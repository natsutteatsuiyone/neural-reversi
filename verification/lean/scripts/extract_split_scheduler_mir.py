"""Extract the owner idle-loop control graph and scheduler call/data flow from pinned Rust MIR."""
import re
from functools import cache

from mir_common import run_extractor_cli, blocks, CALL, GOTO, checked

def edges(term):
    if term in ('return;', 'unreachable;') or re.fullmatch(r'_\d+ = (?:panic|panic_fmt)\(.+\) -> bb\d+;', term):
        return []
    if m := re.fullmatch(GOTO, term):
        return [int(m[1] or m[2])]
    if m := re.search(r' -> \[(?:return|success): bb(\d+), unwind: bb\d+\];$', term):
        return [int(m[1])]
    if term.startswith('switchInt('):
        return [int(v) for v in re.findall(r': bb(\d+)', term)]
    raise ValueError(term)

class Body:
    def __init__(self, dump, name, roots):
        pattern = '*bitset*-' if name in ('set','reset','none','clear','count') else '*-'
        files = list(dump.glob(pattern + name + '.SimplifyCfg-initial.after.mir'))
        checked(len(files) == 1, f'{name}: missing or ambiguous body')
        self.cfg = blocks(files[0].read_text())
        self.defs = {}
        self.roots = dict(enumerate(roots, 1))
        self.marks = {}
        for b in self.cfg.values():
            for s in b:
                if m := re.fullmatch(r'(_\d+) = (.+?)(?: -> .+)?;', s):
                    self.defs.setdefault(m[1],set()).add(m[2])

    @cache
    def val(self, x):
        if m := re.fullmatch(r'(?:move |copy |&mut |&)?(.+)', x):
            x = m[1]
        if m := re.fullmatch(r'\(\*(_\d+)\)', x):
            return self.val(m[1])
        if re.fullmatch(r'_\d+',x):
            if int(x[1:]) in self.roots: return self.roots[int(x[1:])]
            opts = self.defs.get(x, set())
            checked(len(opts) == 1, f'ambiguous {x}: {opts}')
            return self.val(next(iter(opts)))
        if m := re.fullmatch(r'\(\(\*(_\d+)\)\.(\d+): .+\)(?:\[(_\d+)\])?',x):
            field = ('field',self.val(m[1]),int(m[2]))
            return ('index',field,self.val(m[3])) if m[3] else field
        if m := re.fullmatch(r'\((_\d+)\.(\d+): .+\)',x):
            v=self.val(m[1]); j=int(m[2])
            return v[j+1] if isinstance(v,tuple) and v[0]=='tuple' else ('field',v,j)
        if m := re.fullmatch(r'\(\((_\d+) as Some\)\.0: .+\)',x):
            return ('unSome',self.val(m[1]))
        if m := re.fullmatch(r'\((move .+)\)',x):
            return ('tuple',*(self.val(a) for a in m[1].split(', ')))
        if m := re.fullmatch(r'([^()]+)\((.*)\)',x):
            args=tuple(self.val(a) for a in m[2].split(', ') if a)
            if m[1].endswith((' as Deref>::deref',' as Clone>::clone')): return args[0]
            if m[1].startswith('Option::<Arc<SplitPoint>>::expect'): return ('unSome',args[0])
            return (m[1],*args)
        return x

    def call(self, name, args, tag=None):
        sites=[]
        for pc,b in self.cfg.items():
            if (m:=re.fullmatch(CALL,b[-1])) and m[2]==name:
                got=[self.val(op) for op in m[3].split(', ') if op]
                checked(got == args, f'{name} actual {got} expected {args}')
                sites.append(pc)
                if tag: self.marks[pc]=[tag]
        checked(len(sites)==1, f'{name} sites {sites}')
        return sites[0]

    def paths(self, start=0):
        @cache
        def visit(i):
            term=self.cfg[i][-1]
            if term=='return;': return {tuple(self.marks.get(i,[]))}
            return {(*self.marks.get(i,[]),*suffix) for j in edges(term) for suffix in visit(j)}
        return visit(start)

    def projection(self):
        found=self.paths()
        checked(len(found)==1,f'normal paths disagree {found}')
        return list(found.pop())

def owner_fields(dump):
    thr='threading::thread::Thread::'
    fld=lambda x,n:('field',x,n)
    relax='std::sync::atomic::Ordering::Relaxed'
    s=Body(dump,'split',['thread','ctx','req'])
    slot=('index',fld('thread',16),('Atomic::<usize>::load',fld('thread',15),relax))
    s.call('Atomic::<usize>::load',[fld('thread',15),relax])
    s.call(thr+'initialize_split_point',['thread',slot,'ctx','req'],'init')
    s.call(thr+'idle_loop',['thread'],'idle')
    s.call(thr+'finalize_split_point',['thread',slot],'finalize')
    receiver=('SplitPoint::state',slot)
    s.call('SplitPoint::take_counters_after_finished',[slot])
    s.call('SplitPointState::best_score',[receiver],'best')
    s.call('SplitPointState::best_move',[receiver],'picked')
    checked(s.val('_0')==('tuple',('SplitPointState::best_score',receiver),
            ('SplitPointState::best_move',receiver),('SplitPoint::take_counters_after_finished',slot)),
            'split return operands')
    for pc, block in s.cfg.items():
        if block[-1] == 'return;':
            s.marks.setdefault(pc, []).append('returned')
    owner=s.projection()
    g=Body(dump,'initialize_split_point',['thread','slot','ctx','req'])
    st=('SplitPoint::state','slot')
    g.call('AtomicBitSet::clear',[fld(st,6)],'clear')
    g.call('AtomicBitSet::set',[fld(st,6),fld('thread',3)],'mark')
    g.call('SplitPointState::set_best_score',[st,fld('req',3)],'seed')
    g.call('SplitPointState::set_best_move',[st,fld('req',4)],'seedMove')
    g.call('SplitPointState::set_alpha',[st,fld('req',1)],'seedAlpha')
    active=(thr+'active_split_point_mut','thread')
    assign=[pc for pc,b in g.cfg.items() for l in b if (m:=re.fullmatch(r'\(\*(_\d+)\) = move (_\d+);',l))
            and g.val(m[1])==active and g.val(m[2])==('Option::<Arc<SplitPoint>>::Some','slot')]
    checked(len(assign)==1,'active assignment')
    g.marks[assign[0]]=['assign']
    init=g.projection()
    return owner, init

def render(dump, repo):
    src=repo/'crates/reversi-core/src'
    def fields(path, name, expected):
        text=re.sub(r'//[^\n]*','',(src/path).read_text(encoding='utf-8'))
        m=re.search(r'pub struct '+name+r"(?:<'a>)? \{([^}]+)\}",text)
        checked(m is not None, f'missing {name}')
        members=re.findall(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:',m[1],re.M)
        checked(all(members[i]==field for i,field in expected.items()), f'{name} field map')
    fields('search/threading/thread.rs','Thread',{3:'idx',15:'split_points_size',16:'split_points',17:'active_split_point',21:'exit'})
    fields('search/threading/thread.rs','SplitRequest',{1:'alpha',3:'best_score',4:'best_move'})
    sp_file='search/threading/split_point.rs'
    fields(sp_file,'SplitPoint',{1:'state'})
    fields(sp_file,'SplitPointState',{2:'best_score',6:'helpers_mask'})
    fields(sp_file,'SplitPointTask',{9:'node_type'})
    fields('util/bitset.rs','AtomicBitSet',{0:'data'})
    node=(src/'search/node_type.rs').read_text(encoding='utf-8')
    for trait, pv, root in [('NonPV','false','false'),('PV','true','false'),('Root','true','true')]:
        checked(re.search('impl NodeType for '+trait+r' \{\s*const PV_NODE: bool = '+pv+
                          r';\s*const ROOT_NODE: bool = '+root+';',node), trait+' worker specialization')
    thr='threading::thread::Thread::'
    fld=lambda x,n:('field',x,n)
    acq='std::sync::atomic::Ordering::Acquire'
    relax='std::sync::atomic::Ordering::Relaxed'
    rel='std::sync::atomic::Ordering::Release'
    owner,init=owner_fields(dump)
    getter='SyncUnsafeCell::<Option<Arc<SplitPoint>>>::get'
    for method in ['active_split_point','active_split_point_mut']:
        checked(Body(dump,method,['thread']).val('_0')==(getter,fld('thread',17)),method)
    checked(Body(dump,'state',['slot']).val('_0')==fld('slot',1),'state pointer')
    h=Body(dump,'idle_loop',['thread'])
    sp=('unSome',(thr+'active_split_point','thread'))
    mask=fld(('SplitPoint::state',sp),6)
    guarded={}
    for pc,b in h.cfg.items():
        if m:=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',b[-1]):
            # Only trace guards directly consuming the two relevant atomic/bitset calls.
            rhs=h.defs.get(m[1],set())
            if not any(v.startswith(('Atomic::<bool>::load','AtomicBitSet::none')) for v in rhs): continue
            origin=h.val(m[1])
            if origin==('Atomic::<bool>::load',fld('thread',21),acq): guarded[pc]='shutdown'
            if origin==('AtomicBitSet::none',mask): guarded[pc]='mask'
    # The shutdown check inside condvar wait is allowed too; its true edge need not itself return.
    checked(set(guarded.values())=={'shutdown','mask'},'missing owner guard')
    none=Body(dump,'none',['mask'])
    checked(none.val('_0')==('Eq',('Atomic::<u64>::load',fld('mask',0),acq),'const 0_u64'),'acquire/zero')
    best=Body(dump,'best_score',['state'])
    checked(best.val('_0')==('Score::from_raw',('Atomic::<i32>::load',fld('state',2),relax)),'best load')
    raw=Body(dump,'from_raw',['score'])
    checked(raw.val('_0')==('Score','score') and raw.paths()=={()},'Score::from_raw result')
    seed=Body(dump,'set_best_score',['slotState','value'])
    seed.call('Atomic::<i32>::store',[fld('slotState',2),('Score::value','value'),relax],'seed')
    checked(seed.projection()==['seed'],'initial score store')
    set_body=Body(dump,'set',['mask','worker'])
    bit=('Shl','const 1_u64','worker')
    set_body.call('Atomic::<u64>::fetch_or',[fld('mask',0),bit,relax],'mark')
    reset=Body(dump,'reset',['mask','worker'])
    reset.call('Atomic::<u64>::fetch_and',[fld('mask',0),('Not',bit),rel],'release')
    checked(set_body.projection()==['mark'] and reset.projection()==['release'],'bit actions')
    w=Body(dump,'search_active_split_point',['thread'])
    task=('SplitPoint::task',sp); it=('SplitPoint::move_iter',sp)
    w.call(thr+'dispatch_search',['thread',("SearchContext::<'_>::from_split_task",task),task,it,sp],'dispatch')
    w.call('AtomicBitSet::reset',[mask,fld('thread',3)],'release')
    worker=w.projection()
    d=Body(dump,'dispatch_search',['thread','ctx','task','moves','slot'])
    for typ in ['PV','Root','NonPV']:
        d.call(f'search_split_point::<{typ}>',['ctx','task','moves','thread','slot'],typ)
    term=d.cfg[0][-1]
    m=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), 1: bb(\d+), 2: bb(\d+), otherwise: bb(\d+)\];',term)
    checked(m and d.val(m[1])==('discriminant',fld('task',9)),'node-type discriminant')
    modes=[d.paths(int(j)) for j in m.groups()[1:]]
    checked(all(len(p)==1 and len(next(iter(p)))==1 for p in modes[:3]) and not modes[3],'dispatch paths')
    n=max(h.cfg)+1
    allowed={0}; stack=[0]
    while stack:
        i=stack.pop(); dst=edges(h.cfg[i][-1])
        for j in (dst[:1] if i in guarded else dst):
            if j not in allowed: stack.append(j);allowed.add(j)
    def label(i):
        if i not in h.cfg: return '.any []'
        term=h.cfg[i][-1];dst=edges(term)
        if i in guarded: return f'.{guarded[i]} {dst[0]} {dst[1]}'
        if term=='return;': return '.done'
        return '.any ['+', '.join(map(str,dst))+']'
    def items(xs): return '['+', '.join('.'+x for x in xs)+']'
    return '\n'.join(['import NeuralReversi.PvsParallel.SplitSchedulerMir.Syntax','',
        'namespace NeuralReversi.PvsParallel.SplitSchedulerMir','',
        'def ownerPhases : List Phase := '+items(owner),
        'def initPhases : List Phase := '+items(init),
        'def workerPhases : List Phase := '+items(worker),
        'def dispatchModes : List Mode := '+items([next(iter(p))[0] for p in modes[:3]]),'',
        f'def idle : Fin {n} → Op {n} := fun pc => #[', ',\n'.join('  '+label(i) for i in range(n)),'][pc.val]!', '',
        f'def awaiting (pc : Fin {n}) : Bool := #[', '  '+', '.join(str(i in allowed).lower() for i in range(n)),'][pc.val]!', '',
        'end NeuralReversi.PvsParallel.SplitSchedulerMir',''])

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="util::bitset|search::threading::thread|search::threading::split_point|from_raw",
        check_message="scheduler MIR changed",
    )

if __name__=='__main__': main()
