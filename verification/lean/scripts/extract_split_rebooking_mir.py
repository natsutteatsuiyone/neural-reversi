"""Extract the active-parent nested-split guard and the mask-cap gate at both booking sites."""
import re

from extract_split_scheduler_mir import Body, edges
from mir_common import run_extractor_cli, blocks, CALL, checked

THREAD = 'threading::thread::Thread::'
CAP = 'const search::threading::split_point::MAX_THREADS_PER_SPLITPOINT'
RELAXED = 'std::sync::atomic::Ordering::Relaxed'


def nested(dump,limit):
    body=Body(dump,'can_split',['thread'])
    checked(body.defs['_12']=={'&(((*_9) as Some).0: std::sync::Arc<search::threading::split_point::SplitPoint>)'},
            'active parent binding')
    slot='slot'; state=('SplitPoint::state',slot)
    body.roots[12]=slot
    getters={body.call('SplitPointState::all_helpers_searching',[state]):'opened',
             body.call('AtomicBitSet::count',[('field',state,6)]):'count',
             body.call('Atomic::<usize>::load',[('field','thread',15),RELAXED]):'depth'}
    active=body.call(THREAD+'active_split_point',['thread'])
    cap_text='cap'
    def walk(pc,env):
        env=dict(env); lines=body.cfg[pc]; term=lines[-1]
        def val(operand):
            if m:=re.fullmatch(r'(?:copy|move) (_\d+)',operand): return env[m[1]]
            if operand==CAP: return cap_text
            if operand=='const search::threading::thread::MAX_SPLITPOINTS_PER_THREAD':return str(limit)
            if m:=re.fullmatch(r'const (\d+)_u32',operand):return m[1]
            if operand in ['const true','const false']:return operand[6:]
            if operand=='copy ((*_1).11: usize)':return 'pool'
            if m:=re.fullmatch(r'move (_\d+) as u32 \(IntToInt\)',operand):
                checked(env[m[1]]=='pool','pool cast');return 'pool'
            if m:=re.fullmatch(r'(Le|Gt|Eq|Lt)\((.+), (.+)\)',operand):
                a,b=val(m[2]),val(m[3])
                return 'decide ('+a+' '+{'Le':'≤','Gt':'>','Eq':'=','Lt':'<'}[m[1]]+' '+b+')'
            if operand.startswith('discriminant('):
                checked(operand=='discriminant((*_9))' and env['_9']=='activeParent','parent discriminant')
                return 'true'
            return 'unmodeled'
        for line in lines[:-1]:
            if m:=re.fullmatch(r'(_\d+) = (.+);',line):env[m[1]]=val(m[2])
        if m:=re.fullmatch(CALL,term):
            env[m[1]]=getters.get(pc,'activeParent' if pc==active else 'unmodeled')
            return walk(int(m[4]),env)
        if m:=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[(0|1): bb(\d+), otherwise: bb(\d+)\];',term):
            cond=env[m[1]]
            checked(cond!='unmodeled','unmodeled branching condition')
            yes,no=(m[4],m[3]) if m[2]=='0' else (m[3],m[4])
            if cond in ('true','false'):return walk(int(yes if cond=='true' else no),env)
            on,off=walk(int(yes),env),walk(int(no),env)
            if on==off:return on
            return '(if '+cond+' then '+on+' else '+off+')'
        if term=='return;':
            checked(env['_0']!='unmodeled','unknown result')
            return env['_0']
        dst,=edges(term)
        return walk(dst,env)
    return walk(0,{})


def capacity_at_book(dump,late):
    name='try_late_join' if late else 'assign_helpers_to_split_point'
    body=Body(dump,name,['thread'] if late else ['pool','slot'])
    if late:
        checked(body.defs['_72']=={'copy ((_7 as Some).0: &std::sync::Arc<search::threading::split_point::SplitPoint>)'},
                'late join candidate')
        body.roots[72]='slot'
    candidate='thread' if late else body.val('_17')
    state=('SplitPoint::state','slot')
    book=body.call(THREAD+'book_into',[candidate,'slot',state])
    lock=-1; drops=set()
    if late:
        lock=body.call('SplitPoint::lock',['slot'])
        reg=re.fullmatch(CALL,body.cfg[lock][-1])[1]
        drops={pc for pc,ls in body.cfg.items() if ls[-1].startswith(f'drop({reg}) ->')}
    guards={}
    for pc,ls in body.cfg.items():
        if not (m:=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',ls[-1])):continue
        if not any(CAP in d for d in body.defs.get(m[1],set())):continue
        try:origin=body.val(m[1])
        except ValueError:continue
        count=('AtomicBitSet::count',('field',state,6))
        if origin==('Lt',count,CAP):guards[pc]=1
        if origin==('Ge',count,CAP):guards[pc]=0
    todo=[(0,not late,False)];seen=set();credits=set()
    while todo:
        item=todo.pop()
        if item in seen:continue
        seen.add(item)
        pc,held,credit=item
        if pc==lock:held=True
        if pc in drops:held,credit=False,False
        if pc==book:credits.add(credit and held);credit=False
        for i,dst in enumerate(edges(body.cfg[pc][-1])):
            new_credit=(i==guards[pc] and held) if pc in guards else credit
            todo.append((dst,held,new_credit))
    checked(len(credits)==1,'inconsistent booking checks')
    return 'decide (count < cap)' if credits.pop() else 'true'


def render(dump,repo):
    path=repo/'crates/reversi-core/src/search/threading'
    def num(file,name):
        matches=re.findall(r'(?:^| )const '+name+r': (?:u32|usize) = (\d+);',(path/file).read_text(encoding='utf8'),re.M)
        checked(len(matches)==1,'bound '+name)
        return int(matches[0])
    cap=num('split_point.rs','MAX_THREADS_PER_SPLITPOINT')
    limit=num('thread.rs','MAX_SPLITPOINTS_PER_THREAD')
    cnt=Body(dump,'count',['mask'])
    checked(cnt.val('_0')==('core::num::<impl u64>::count_ones',('Atomic::<u64>::load',('field','mask',0),RELAXED)),
            'count_ones of relaxed mask load')
    cfg=blocks((dump/'reversi_core.search-pvs-search.SimplifyCfg-initial.after.mir').read_text())
    call=[(pc,m) for pc,ls in cfg.items() if (m:=re.fullmatch(CALL,ls[-1]))]
    guard,=[(pc,m) for pc,m in call if m[2]==THREAD+'can_split']
    split,=[(pc,m) for pc,m in call if m[2]==THREAD+'split']
    pc=int(guard[1][4])
    check=re.fullmatch(r'switchInt\(move '+guard[1][1]+r'\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',cfg[pc][-1])
    checked(check and int(check[2])==split[0],'nested call guard')
    entrants=[i for i,ls in cfg.items() if i!=pc and (' bb'+str(split[0])+',' in ls[-1] or
              ' bb'+str(split[0])+']' in ls[-1] or ' bb'+str(split[0])+';' in ls[-1])]
    checked(not entrants,'bypass to split')
    return '\n'.join(['module', 'public import NeuralReversi.Threading.AtomicBitSet','','@[expose] public section', '', 'namespace NeuralReversi.PvsParallel.SplitRebookingMir','',
        'def cap : Nat := '+str(cap),'','def nested (pool depth : Nat) (opened : Bool) (count : Nat) : Bool :=',
        '  '+nested(dump,limit),'','def room (late : Bool) (count : Nat) : Bool :=',
        '  if late then '+capacity_at_book(dump,True)+' else '+capacity_at_book(dump,False),'',
        'end NeuralReversi.PvsParallel.SplitRebookingMir',''])

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="can_split|assign_helpers_to_split_point|try_late_join|util::bitset|search::pvs::search",
        check_message="nested scheduling gates changed",
    )

if __name__=='__main__':main()
