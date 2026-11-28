"""Extract incoming critical call edges and searching/failed atomic accesses from all crate MIR bodies."""
import itertools
import re

from extract_split_scheduler_mir import edges
from mir_common import PASS, run_extractor_cli, CALL, blocks, checked, iter_mir_texts

SUFFIX='.SimplifyCfg-initial.after.mir'
TH='reversi_core.search-threading-thread-{impl#1}-'
PL='reversi_core.search-threading-pool-{impl#0}-'
KINDS={
    TH+'search_active_split_point':'work', TH+'run_helper_job':'helper',
    TH+'run_helper_job-{impl#0}-drop':'helperDrop', TH+'idle_loop':'idle',
    TH+'main_thread_loop':'mainLoop', TH+'split':'split',
    TH+'abandon_active_split':'abandon', TH+'detach_split':'detach',
    PL+'create_main_thread-{closure#0}':'mainEntry',
    PL+'create_worker_threads-{closure#0}':'workerEntry',
    TH+'book_into':'book',TH+'book_helper_job':'bookJob',TH+'finalize_split_point':'finalize',
}
METHODS={'search_active_split_point':'work','run_helper_job':'helper','idle_loop':'idle',
         'main_thread_loop':'mainLoop','split':'split','abandon_active_split':'abandon','detach_split':'detach'}
FIELD_PAT=re.compile(r'(_\d+) = &\(.+?\.(19|20): (util::align::Align64<std::sync::atomic::Atomic<bool>>|std::sync::atomic::Atomic<bool>)\);')
ANY_FIELD=re.compile(r'\.(?:19: util::align::Align64<std::sync::atomic::Atomic<bool>>|20: std::sync::atomic::Atomic<bool>)')


def helpful_guard(cfg, inputs):
    searching, failed, has_slot, has_deepest, tested = inputs
    env={'_1':'worker','_2':'requested'}
    pc=0
    for _ in range(len(cfg)+1):
        body=cfg[pc]
        for line in body[:-1]:
            if line.startswith(('Storage','FakeRead','PlaceMention')):
                continue
            left,right=line.removesuffix(';').split(' = ',1)
            if right in ['const true','const false','const ()']:
                env[left]={'const true':True,'const false':False,'const ()':None}[right]
            elif right.startswith('std::sync::atomic::Ordering::'):
                env[left]=right.split('::')[-1]
            elif m:=re.fullmatch(r'&\(\*(_\d+)\)|copy (_\d+)',right):
                env[left]=env[m[1] or m[2]]
            elif m:=re.fullmatch(r'&\(\(\*(_\d+)\)\.(\d+): .+\)',right):
                env[left]=(env[m[1]],int(m[2]))
            elif m:=re.fullmatch(r'Eq\(move (_\d+), const 0_usize\)',right):
                env[left]=not env[m[1]]
            elif m:=re.fullmatch(r'discriminant\((_\d+)\)',right):
                checked(env[m[1]] in ('some-deepest','none'),'unexpected Option discriminator')
                env[left]=int(env[m[1]]=='some-deepest')
            elif m:=re.fullmatch(r'copy \(\((_\d+) as Some\)\.0:.+\)',right):
                checked(env[m[1]]=='some-deepest','invalid helpful-owner downcast')
                env[left]='deepest'
            else:raise ValueError('unrecognized helpful-owner rvalue: '+right)
        term=body[-1]
        if term=='return;':return env['_0']
        if m:=re.fullmatch(CALL,term):
            dest,fun,operands,nxt=m.groups()
            args=[env[x.removeprefix('move ').removeprefix('copy ')] for x in operands.split(', ')]
            if fun.endswith(' as Deref>::deref'):
                env[dest],=args
            elif fun=='Atomic::<bool>::load':
                env[dest]={ (('worker',19),'Acquire'):searching,(('worker',20),'Relaxed'):failed }[tuple(args)]
            elif fun=='Atomic::<usize>::load':
                checked(args==[('worker',15),'Acquire'],'wrong split depth word or ordering');env[dest]=has_slot
            elif fun.endswith('Thread::deepest_split_point'):
                checked(args==['worker',has_slot],'wrong deepest slot operands')
                env[dest]='some-deepest' if has_deepest else 'none'
            elif fun=='SplitPoint::state':
                source,=args;env[dest]=('state',source)
            elif fun=='SplitPointState::owner_thread_idx':
                source,=args;env[dest]=('owner',source)
            elif fun=='AtomicBitSet::test':
                checked(args==[(('state','deepest'),6),('owner',('state','requested'))],
                        'can_join tests a different mask or a different owner')
                env[dest]=tested
            else:raise ValueError('unrecognized helpful-owner call: '+fun)
            pc=int(nxt)
        elif m:=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];',term):
            pc=int(m[3] if env[m[1]]==int(m[2]) else m[4])
        else:
            successor,=edges(term);pc=successor
    raise ValueError('can_join did not return')


def inspect_fields(cfg, name):
    local={}
    checked_ops=[]
    aliases={}
    # These thread-field aliases are SSA in the pinned bodies. Reject ambiguous definitions.
    for pc,ls in cfg.items():
        for l in ls:
            m=FIELD_PAT.fullmatch(l)
            if m: local[m[1]] = 'searching' if m[2]=='19' else 'failed'
            m=re.fullmatch(r'(_\d+) = (.+?)(?: -> .+)?;',l)
            if m:aliases.setdefault(m[1],set()).add(m[2])
    def field(v, seen=frozenset()):
        if v in local:return local[v]
        if v in seen:return None
        s=aliases.get(v,set())
        if len(s)!=1:return None
        rhs=next(iter(s))
        if m:=re.fullmatch(r'&\(\*(_\d+)\)|<Align64<Atomic<bool>> as Deref>::deref\(move (_\d+)\)',rhs):
            return field(m[1] or m[2],seen|{v})
        return None
    used=set()
    for ls in cfg.values():
        for line in ls:
            if line.startswith(('Storage','FakeRead','PlaceMention')):continue
            if ANY_FIELD.search(line):checked(FIELD_PAT.fullmatch(line),'non-reference access: '+line)
            rhs=line.split(' = ',1)[-1]
            for var in re.findall(r'\b_\d+\b',rhs):
                if field(var) is None:continue
                allowed=rf'&\(\*{var}\);|<Align64<Atomic<bool>> as Deref>::deref\(move {var}\) -> .+|Atomic::<bool>::(?:load|store)\((?:move|copy) {var}, [^)]*\) -> .+'
                checked(re.fullmatch(allowed,rhs),'escaped searching/failed reference: '+line)
    for pc,ls in cfg.items():
        term=ls[-1];m=re.fullmatch(CALL,term)
        if not m:continue
        if m[2].endswith(' as Deref>::deref'):continue
        arg=m[3].split(', ')
        if not arg or not (v:=re.fullmatch(r'(?:copy|move) (_\d+)',arg[0])):continue
        k=field(v[1])
        if k is None:continue
        checked(m[2] in ['Atomic::<bool>::load','Atomic::<bool>::store'], 'unhandled atomic operation '+name+': '+term)
        method=m[2].split('::')[-1]
        checked(len(arg) == (3 if method=='store' else 2), 'atomic arity')
        order=aliases[arg[-1].split()[-1]]
        checked(len(order)==1,'atomic ordering ambiguity')
        ordering=next(iter(order)).split('::')[-1]
        value=arg[1] if method=='store' else None
        checked(value in (None,'const true','const false'),'dynamic searching/failed write')
        checked_ops.append((pc,k,method,value,ordering))
        used.add(k)
    checked(set(local.values()) <= used,f'untraced field reference: {name}')
    return checked_ops


def render(dump):
    names=set(); callees={s:set() for s in METHODS.values()}
    callees.update({k:set() for k in ['mainEntry','workerEntry','helperDrop']})
    writers=set(); failed=[]; effects={}; count=0
    critical=re.compile(r'\b(?:threading::thread::)?Thread::('+'|'.join(METHODS)+r')\b')
    for f, source in iter_mir_texts(dump, SUFFIX):
        name=f.name[:-len(SUFFIX)]
        count+=1;names.add(name)
        caller=KINDS.get(name,'other')
        if 'Thread::' in source:
            for line in source.splitlines():
                if not line.startswith('        '):continue
                if m:=critical.search(line):
                    checked(re.search(r'\bThread::'+m[1]+r'\(',line) and ' -> [' in line,
                            'non-direct protected method reference: '+line)
                    callees[METHODS[m[1]]].add(caller)
        if 'FinishJob::<' in source:
            checked(name==TH+'run_helper_job','private FinishJob allocated elsewhere')
            guard=re.search(r'(_\d+) = FinishJob::<\x27_>\(move _\d+\);',source)
            checked(guard,'unrecognized destructor construction')
            local=guard[1]
            for line in source.splitlines():
                if not line.startswith('        ') or not re.search(r'\b'+local+r'\b',line):continue
                checked(line.strip().startswith(('debug ', 'Storage', 'FakeRead(',f'{local} = FinishJob::',f'drop({local}) ->')),
                        'local FinishJob guard escaped: '+line)
            callees['helperDrop'].add('helper')
        if name in (PL+'create_main_thread',PL+'create_worker_threads'):
            m=re.search(r'(_\d+) = (\{closure@[^}]+\}) \{ thread: (?:move|copy) _\d+(?:, receiver: move _\d+)? \};',source)
            checked(m and re.search(r'spawn::<'+re.escape(m[2])+r', \(\)>\(move '+m[1]+r'\)',source),
                    'thread startup closure is not passed to std::thread::spawn')
        if ANY_FIELD.search(source):
            cfg=blocks(source)
            raw=inspect_fields(cfg,name)
            effects[name]=(cfg,raw)
            for pc,field,op,value,order in raw:
                if op!='store':continue
                if field=='searching':
                    checked(order=='Release' and caller!='other','unexpected searching store '+name)
                    writers.add((caller,value=='const true'))
                else:
                    checked(value=='const true' and order=='Relaxed','failed flag cleared after initialization')
                    failed.append((caller,pc))
    checked(count>0 and set(KINDS)<=names,'incomplete crate projection')
    checked(set(callees['idle'])=={'split','workerEntry'} and callees['mainLoop']=={'mainEntry'},'thread entrypoints')
    checked({a for a,_ in failed}=={'helperDrop','abandon'},'failure writers')
    # Scope data for pool shutdown and faulted workers.
    cfg,raw=effects[TH+'abandon_active_split']
    failPc,=[pc for pc,f,op,val,order in raw if f=='failed' and op=='store']
    detached,=[pc for pc,ls in cfg.items() if '::detach_split(' in ls[-1]]
    todo=[(0,False)];seen=set()
    while todo:
        pc,failedBefore=todo.pop()
        if (pc,failedBefore) in seen:continue
        seen.add((pc,failedBefore)); failedBefore|=(pc==failPc)
        if pc==detached:checked(failedBefore,'abandon detach can publish false before failed=true')
        todo.extend((dst,failedBefore) for dst in edges(cfg[pc][-1]))
    cfg,raw=effects[TH+'run_helper_job-{impl#0}-drop']
    panic,=[pc for pc,ls in cfg.items() if re.search(r' = (?:(?:std::)?thread::)?panicking\(\) ->', ls[-1])]
    chk,=edges(cfg[panic][-1]);cond=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',cfg[chk][-1])
    checked(cond and re.fullmatch(CALL,cfg[panic][-1])[1]==cond[1],'helper panic branch')
    for branch,expected in [(int(cond[2]),False),(int(cond[3]),True)]:
        todo=[(branch,False)]
        while todo:
            pc,quarantined=todo.pop()
            quarantined|=any(p==pc and k=='failed' and op=='store' for p,k,op,val,order in raw)
            if any(p==pc and k=='searching' and op=='store' for p,k,op,val,order in raw):
                checked(quarantined==expected,'helper completion quarantine order')
            todo.extend((dst,quarantined) for dst in edges(cfg[pc][-1]))
    cfg,raw=effects[TH+'can_join']
    for args in itertools.product((False,True),repeat=5):
        searching,fault,slot,deep,bit=args
        checked(helpful_guard(cfg,args)==(not searching and not fault and (not slot or deep and bit)),
                'helpful-owner decision changed')
    reject_busy=[]
    for pc,field,op,value,order in raw:
        checked(op=='load' and (field,order) in [('searching','Acquire'),('failed','Relaxed')],'can_join flags')
        reject_busy.append(field)
        branch,=edges(cfg[pc][-1]);t=cfg[branch][-1]
        chk=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',t)
        checked(chk and re.fullmatch(CALL,cfg[pc][-1])[1]==chk[1],'bad protected guard')
        todo=[(int(chk[3]),None)]
        while todo:
            n,v=todo.pop()
            for l in cfg[n]:
                if l.startswith('_0 = '):v=l
            if cfg[n][-1]=='return;':checked(v=='_0 = const false;','busy/failed worker accepted')
            todo.extend((j,v) for j in edges(cfg[n][-1]))
    ctor=(dump/(TH+'new'+SUFFIX)).read_text()
    cfg=blocks(ctor);s=next(l for ls in cfg.values() for l in ls if 'Thread { mutex_for_sleep_condition:' in l)
    for k in ['failed','searching']:
        var=re.search(k+r': move (_\d+),',s)[1]
        if k=='searching':var=re.search(re.escape(var)+r' = Align64::<Atomic<bool>>\(move (_\d+)\);',ctor)[1]
        checked(re.search(re.escape(var)+r' = Atomic::<bool>::new\(const false\)',ctor),'nonzero worker initializer')
    out=['module', 'public import NeuralReversi.PvsParallel.WorkerFrameMir.Syntax','','@[expose] public section', '', 'namespace NeuralReversi.PvsParallel.WorkerFrameMir','',
         'def callers : Kind → Option (List Kind)']
    for c,parents in sorted(callees.items()):
        out.append('  | .'+c+' => some ['+', '.join('.'+p for p in sorted(parents))+']')
    out+=['  | _ => none','','def searchingStores : List (Kind × Bool) := [',
        ',\n'.join('  (.'+k+', '+str(b).lower()+')' for k,b in sorted(writers)),']','',
        'def failedStores : List Kind := ['+', '.join('.'+k for k in sorted({k for k,_ in failed}))+']','',
        'def joinCheck (searching failed : Bool) : Bool := '+' && '.join('!'+f for f in reject_busy),'',
        'def canJoin (searching failed hasSlot deepest tested : Bool) : Bool :=',
        '  joinCheck searching failed && (!hasSlot || (deepest && tested))','',
        'end NeuralReversi.PvsParallel.WorkerFrameMir','']
    return '\n'.join(out)

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters=PASS,
        takes_repo=False,
        allow_output=True,
        check_message="worker frame graph changed",
    )
if __name__=='__main__':main()
