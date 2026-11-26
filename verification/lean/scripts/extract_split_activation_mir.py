"""Extract necessary protected-branch conditions at each scheduler booking site."""
import re

from extract_split_scheduler_mir import Body, edges
from mir_common import run_extractor_cli, CALL, checked

THREAD = 'threading::thread::Thread::'
ACQUIRE = 'std::sync::atomic::Ordering::Acquire'
RELAXED = 'std::sync::atomic::Ordering::Relaxed'
RELEASE = 'std::sync::atomic::Ordering::Release'


def field(obj, idx):
    return ('field', obj, idx)


def lock(body, name, obj):
    pc = body.call(name, [obj])
    return pc, re.fullmatch(CALL, body.cfg[pc][-1])[1]


def drop_of(body, local):
    sites = {pc for pc, lines in body.cfg.items() if lines[-1].startswith(f'drop({local}) -> ')}
    checked(sites, f'missing drop of {local}')
    return sites


def at_protected_booking(body, slot, thread, late):
    state = ('SplitPoint::state', slot)
    signal = 'SplitPointState::all_helpers_searching'
    site = body.call(THREAD + 'book_into', [thread, slot, state])
    acquire_thread, th_guard = lock(body, THREAD + 'lock', thread)
    drop_thread = drop_of(body, th_guard)
    acquire_sp, drop_sp = -1, set()
    if late:
        acquire_sp, sp_guard = lock(body, 'SplitPoint::lock', slot)
        drop_sp = drop_of(body, sp_guard)
    guards = {}
    for pc, lines in body.cfg.items():
        m = re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];', lines[-1])
        if not m: continue
        defs = body.defs.get(m[1], set())
        if not any(x.startswith((THREAD+'can_join(',signal+'(')) for x in defs): continue
        # A negative precheck before the selected best_sp exists cannot authorize a booking.
        try:
            actual = body.val(m[1])
        except ValueError:
            continue
        if actual == (THREAD+'can_join', thread, slot):
            guards[pc] = 'idle'
        if actual == (signal, state):
            guards[pc] = 'open'
    seen = set()
    stack = [(0, not late, False, False, False)]
    booking = set()
    while stack:
        old = stack.pop()
        if old in seen: continue
        seen.add(old)
        pc, sp_locked, th_locked, opened, idle = old
        if pc in drop_sp: sp_locked, opened = False, False
        if pc in drop_thread: th_locked, idle = False, False
        if pc == acquire_sp: sp_locked = True
        if pc == acquire_thread: th_locked = True
        if pc == site:
            booking.add((sp_locked, th_locked, opened, idle))
        dst = edges(body.cfg[pc][-1])
        for choice, nxt in enumerate(dst):
            op = (choice == 1 and sp_locked) if guards.get(pc) == 'open' else opened
            free = (choice == 1 and th_locked) if guards.get(pc) == 'idle' else idle
            stack.append((nxt, sp_locked, th_locked, op, free))
    checked(len(booking) == 1, f'booking entered with variable protection: {booking}')
    sp_locked, th_locked, opened, idle = booking.pop()
    checked(sp_locked and th_locked, 'unprotected booking')
    return opened, idle


def idle_authorization(dump):
    body=Body(dump,'idle_loop',['thread'])
    site=body.call(THREAD+'search_active_split_point',['thread'])
    at={m[1]:(m[2],m[3]) for lines in body.cfg.values() if (m:=re.fullmatch(CALL,lines[-1]))}
    def through(local, ending):
        while local not in at:
            definition,=body.defs[local]
            m=re.fullmatch(r'&(?:mut )?(?:\(\*)?(_\d+)\)?',definition)
            checked(m, f'unexpected helper job reference {definition}')
            local=m[1]
        method,arg=at[local]
        checked(method.endswith(ending), 'helper job binding '+method)
        return arg.split()[-1]
    guards={}
    for pc,lines in body.cfg.items():
        term=lines[-1]
        m=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',term)
        if m and body.defs.get(m[1],set()) and all('Atomic::<bool>::load(' in expr for expr in body.defs[m[1]]) \
                and body.val(m[1])==('Atomic::<bool>::load',field('thread',19),ACQUIRE):
            guards[pc]='busy'
        m=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), 1: bb(\d+), otherwise: bb(\d+)\];',term)
        if m:
            definition,=body.defs[m[1]]
            source=re.fullmatch(r'discriminant\((_\d+)\)',definition)
            if not source: continue
            take=at.get(source[1])
            if not take or '>::take' not in take[0]: continue
            ref=through(source[1],'>>::take')
            ref=through(ref,'>::deref_mut')
            ref=through(ref,'>::unwrap')
            ref=through(ref,'>::lock')
            checked(body.val(ref)==field('thread',6),'helper job originated on another thread')
            guards[pc]='job'
    seen=set(); todo=[(0,False,False)]; allowed=set()
    while todo:
        item=todo.pop()
        if item in seen: continue
        seen.add(item)
        pc,busy,empty=item
        if pc==site: allowed.add((busy,empty)); busy,empty=False,False
        for i,j in enumerate(edges(body.cfg[pc][-1])):
            todo.append((j,i==1 if guards.get(pc)=='busy' else busy,i==0 if guards.get(pc)=='job' else empty))
    checked(len(allowed)==1,'ambiguous worker entry')
    return allowed.pop()


def render(dump):
    b = Body(dump, 'book_into', ['thread', 'slot', 'state'])
    b.call('AtomicBitSet::set', [field('state', 6), field('thread', 3)], 'add')
    active = (THREAD+'active_split_point_mut','thread')
    act_at = [pc for pc, lines in b.cfg.items() for item in lines if
        (m := re.fullmatch(r'\(\*(_\d+)\) = move (_\d+);',item)) and b.val(m[1]) == active
        and b.val(m[2]) == ('Option::<Arc<SplitPoint>>::Some','slot')]
    checked(len(act_at) == 1, 'booked slot differs from assigned arc')
    b.marks[act_at[0]] = ['assign']
    b.call('Atomic::<bool>::store', [field('thread',19),'const true',RELEASE], 'wake')
    book = b.projection()

    d = Body(dump, 'detach_split', ['thread','guard'])
    d.call('Option::<Arc<SplitPoint>>::take', [active], 'take')
    d.call('Atomic::<bool>::store',[field('thread',19),'const false',RELEASE],'sleep')
    checked(d.val('_0') == ('Option::<Arc<SplitPoint>>::take',active),'detached slot identity')
    detach = d.projection()

    s = Body(dump,'search_active_split_point',['thread'])
    captured = ('unSome',(THREAD+'active_split_point','thread'))
    split_state = ('SplitPoint::state',captured)
    lp, lg = lock(s,'SplitPoint::lock',captured)
    # The first thread lock snapshots the active job. The second encloses detach and bit reset.
    thread_locks = [(pc,m[1]) for pc, lines in s.cfg.items() if (m:=re.fullmatch(CALL,lines[-1]))
                    and m[2]==THREAD+'lock' and s.val(m[3])=='thread']
    checked(len(thread_locks)==2, 'worker lock sites')
    held = s.call(THREAD+'detach_split',['thread',(THREAD+'lock','thread')], 'detach')
    guard_arg = re.fullmatch(CALL,s.cfg[held][-1])[3].split(', ')[1]
    # Identify the guard by following ordinary reference assignments, preserving both call occurrences.
    local = guard_arg.split()[-1]
    while local not in dict((dst,pc) for pc,dst in thread_locks):
        actual = next(iter(s.defs[local]))
        m = re.fullmatch(r'&(?:\(\*)?(_\d+)\)?', actual)
        checked(m is not None, 'guard pointer origin')
        local = m[1]
    (tp,tg), = [(pc,dst) for pc,dst in thread_locks if dst == local]
    s.marks[lp]=['slotLock']; s.marks[tp]=['threadLock']
    for pc in drop_of(s,tg): s.marks[pc]=['threadUnlock']
    s.call('SplitPointState::set_all_helpers_searching_locked',[split_state,'const false'],'close')
    s.call('AtomicBitSet::reset',[field(split_state,6),field('thread',3)],'remove')
    worker = s.projection()

    init=Body(dump,'initialize_split_point',['thread','slot','ctx','req'])
    sl,sg=lock(init,'SplitPoint::lock','slot')
    init.marks[sl]=['slotLock']
    for pc in drop_of(init,sg): init.marks[pc]=['slotUnlock']
    state=('SplitPoint::state','slot')
    init.call('AtomicBitSet::clear',[field(state,6)],'clear')
    init.call('AtomicBitSet::set',[field(state,6),field('thread',3)],'add')
    init.call('SplitPointState::set_all_helpers_searching_locked',[state,'const true'],'accept')
    assigns=[pc for pc,ls in init.cfg.items() if (m:=re.fullmatch(CALL,ls[-1]))
             and m[2]=='ThreadPool::assign_helpers_to_split_point' and init.val(m[3].split(', ')[1])=='slot']
    checked(len(assigns)==1,'owner helper assignment target')
    init.marks[assigns[0]]=['initialCandidates']
    owner_assign=[pc for pc,lines in init.cfg.items() for item in lines
                 if (m:=re.fullmatch(r'\(\*(_\d+)\) = move (_\d+);',item))
                 and init.val(m[1])==active and init.val(m[2])==('Option::<Arc<SplitPoint>>::Some','slot')]
    checked(len(owner_assign)==1,'owner assignment identity')
    init.marks[owner_assign[0]]=['assign']
    startup=init.paths()
    checked(len(startup)==2,'unexpected number of initialization paths')
    init_program=next(x for x in startup if 'initialCandidates' in x)
    checked(tuple(x for x in init_program if x!='initialCandidates') in startup,'initialization branch loses an operation')

    can=Body(dump,'can_join',['thread','slot'])
    loads=[(pc,m[1]) for pc,ls in can.cfg.items() if (m:=re.fullmatch(CALL,ls[-1]))
           and m[2]=='Atomic::<bool>::load' and can.val(m[3].split(', ')[0])==field('thread',19)
           and can.val(m[3].split(', ')[1])==ACQUIRE]
    checked(len(loads)==1,'searching guard operand')
    load, dst=loads[0]
    switch,=edges(can.cfg[load][-1])
    checked(can.cfg[switch][-1].startswith('switchInt(move '+dst+') -> [0: '),'searching guard polarity')
    todo=[(edges(can.cfg[switch][-1])[1],None)]
    while todo:
        pc,value=todo.pop()
        for line in can.cfg[pc]:
            if line.startswith('_0 = '): value=line
        if can.cfg[pc][-1]=='return;': checked(value=='_0 = const false;','busy worker accepted')
        todo.extend((n,value) for n in edges(can.cfg[pc][-1]))

    initial=Body(dump,'assign_helpers_to_split_point',['pool','slot'])
    candidate=initial.val('_17')
    checked(candidate[0]=='unSome' and 'Iterator>::next' in candidate[1][0] and
            candidate[1][1][1]==field('pool',0),'helper did not come from pool.threads iterator')
    iopen, iid=at_protected_booking(initial,'slot',candidate,False)
    late=Body(dump,'try_late_join',['thread'])
    checked(late.defs['_72']=={'copy ((_7 as Some).0: &std::sync::Arc<search::threading::split_point::SplitPoint>)'},
            'late selected slot binding')
    late.roots[72]='slot'  # fresh name for the captured selected Arc on the protected path
    lopen,lid=at_protected_booking(late,'slot','thread',True)
    read=Body(dump,'all_helpers_searching',['state'])
    checked(read.val('_0')==('Atomic::<bool>::load',field('state',0),RELAXED),'slot flag read')
    flag=Body(dump,'set_all_helpers_searching_locked',['state','flag'])
    flag.call('Atomic::<bool>::store',[field('state',0),'flag',RELAXED],'flag')
    checked(flag.projection()==['flag'],'slot flag write')
    clear=Body(dump,'clear',['mask'])
    clear.call('Atomic::<u64>::store',[field('mask',0),'const 0_u64',RELAXED],'clear')
    add=Body(dump,'set',['mask','worker'])
    bit=('Shl','const 1_u64','worker')
    add.call('Atomic::<u64>::fetch_or',[field('mask',0),bit,RELAXED],'add')
    remove=Body(dump,'reset',['mask','worker'])
    remove.call('Atomic::<u64>::fetch_and',[field('mask',0),('Not',bit),RELEASE],'remove')
    checked([clear.projection(),add.projection(),remove.projection()]==[['clear'],['add'],['remove']],
            'mask primitive operations')
    awake, no_job=idle_authorization(dump)
    def gate(opened,idle):
        return ' && '.join(x for x,y in [('available',opened),('idle',idle)] if y) or 'true'
    def items(xs): return '['+', '.join('.'+x for x in xs)+']'
    return '\n'.join(['import NeuralReversi.PvsParallel.SplitActivationMir.Syntax','',
      'namespace NeuralReversi.PvsParallel.SplitActivationMir','',
      'def initialOps : List Op := '+items(init_program),
      'def enrollOps : List Op := '+items(book),
      'def releaseOps : List Op := '+items(worker),
      'def detachOps : List Op := '+items(detach),'',
      'def gate (late available idle : Bool) : Bool :=',
      '  if late then '+gate(lopen,lid)+' else '+gate(iopen,iid),'',
      'def entersWorker (searching hasHelperJob : Bool) : Bool := '+
          (' && '.join(x for x,required in [('searching',awake),('!hasHelperJob',no_job)] if required) or 'true'),'',
      'end NeuralReversi.PvsParallel.SplitActivationMir',''])

def main():
    filters = (
        'book_into|can_join|detach_split|assign_helpers_to_split_point|try_late_join|'
        'abandon_active_split|helpers_searching|initialize_split_point|'
        'search_active_split_point|idle_loop|util::bitset'
    )
    run_extractor_cli(
        __doc__,
        render,
        filters=filters,
        takes_repo=False,
        check_message="scheduler activation MIR changed",
    )

if __name__=='__main__':
    main()
