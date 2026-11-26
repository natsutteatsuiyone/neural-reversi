"""Extract production slot identities captured, saved, restored and reset on a worker."""
import re

from extract_split_activation_mir import THREAD, field, RELEASE, RELAXED
from extract_split_scheduler_mir import Body,edges
from extract_worker_frame_mir import SUFFIX, TH
from mir_common import PASS, run_extractor_cli, CALL, checked


def render(dump):
    namespace='NeuralReversi.PvsParallel.SlotFrameMir'
    active=(THREAD+'active_split_point','u')
    writable=(THREAD+'active_split_point_mut','u')
    slot=('index',field('u',16),('Atomic::<usize>::load',field('u',15),RELAXED))
    captured=('unSome',active)
    take=(THREAD+'detach_split','u',(THREAD+'lock','u'))
    removed=('unSome',take)
    task=('SplitPoint::task','s')
    saved=('SplitPointTask::new','r',active,'ctx')
    task_cell=('SyncUnsafeCell::<Option<SplitPointTask>>::get',field('s',3))
    def writes(body,place,value,tag):
        sites=[]
        for pc,ls in body.cfg.items():
            for line in ls:
                if m:=re.fullmatch(r'\(\*(_\d+)\) = move (_\d+);',line):
                    checked(body.val(m[1])==place and body.val(m[2])==value,'unexpected indirect store '+line)
                    body.marks[pc]=[tag]
                    sites.append(pc)
        checked(len(sites)==1,'indirect store count')
    # All regular users of these mutators in the crate. The checked Thread and
    # SplitPoint accessors resolve the matching UnsafeCell and field.
    inv={method:set() for method in [THREAD+'active_split_point_mut','SplitPoint::set_task','SplitPoint::clear_task',
                                      'AtomicBitSet::reset','SplitPointState::set_owner_thread_idx']}
    cells={'.17: std::cell::SyncUnsafeCell<std::option::Option<std::sync::Arc<search::threading::split_point::SplitPoint>>>':set(),
           '.3: std::cell::SyncUnsafeCell<std::option::Option<search::threading::split_point::SplitPointTask>>':set(),
           '.16: [std::sync::Arc<search::threading::split_point::SplitPoint>; 8]':set()}
    for f in dump.glob('*'+SUFFIX):
        source=f.read_text(encoding='utf8');name=f.name.removesuffix(SUFFIX)
        for marker,users in cells.items():
            if marker in source:users.add(name)
        for line in source.splitlines():
            if not line.startswith('        '):continue
            for name in inv:
                if name not in line:continue
                m=re.fullmatch(CALL,line.strip())
                checked(m and m[2]==name,'escaped slot mutator: '+line)
                inv[name].add(f.name.removesuffix(SUFFIX))
    expected=[['book_into','detach_split','finalize_split_point','initialize_split_point'],
              ['initialize_split_point'],['finalize_split_point'],['abandon_active_split','search_active_split_point'],['initialize_split_point']]
    for actual,names in zip(inv.values(),expected):
        checked(actual=={TH+n for n in names},'slot writer inventory: '+str(actual))
    sp='reversi_core.search-threading-split_point-{impl#5}-'
    expected=[{TH+'active_split_point',TH+'active_split_point_mut'},
              {sp+'task',sp+'set_task',sp+'clear_task'},{TH+'split',TH+'deepest_split_point-{closure#0}'}]
    checked(list(cells.values())==expected,'direct slot access bypasses captured fields')
    new=(dump/(TH+'new'+SUFFIX)).read_text(encoding='utf8')
    initbank=re.search(r'(_\d+) = std::array::from_fn::<Arc<SplitPoint>, (\d+), (\{closure@[^}]+\})>\(move (_\d+)\)',new)
    checked(initbank and re.search(re.escape(initbank[4])+r' = '+re.escape(initbank[3])+r';',new),'slot-bank allocator')
    named=re.search(r'Thread \{ [^\n]*split_points: move (_\d+),',new)
    checked(named and (named[1]==initbank[1] or re.search(re.escape(named[1])+r' = move '+re.escape(initbank[1])+r';',new)),'new bank not retained by its thread')
    allocated=(dump/(TH+'new-{closure#0}'+SUFFIX)).read_text(encoding='utf8')
    checked(re.search(r'_0 = Arc::<SplitPoint>::new\(move _\d+\) ->',allocated),'aliased owner banks')
    tasknew=(dump/('reversi_core.search-threading-split_point-{impl#3}-new'+SUFFIX)).read_text(encoding='utf8')
    rec=re.search(r'_0 = SplitPointTask \{ ([^}]+) \};',tasknew)
    fields=rec[1].split(', ')
    parent,value=fields[10].split(': ')
    checked(parent=='parent_split_point' and re.search(re.escape(value.split()[-1])+r' = move _2;',tasknew),'task parent field')
    for fn in ['active_split_point','active_split_point_mut']:
        checked(Body(dump,fn,['u']).val('_0')==('SyncUnsafeCell::<Option<Arc<SplitPoint>>>::get',field('u',17)),fn)
    for fn,value in [('set_task',('Option::<SplitPointTask>::Some','task')),('clear_task','Option::<SplitPointTask>::None')]:
        b=Body(dump,fn,['s','task']);writes(b,task_cell,value,'write')
        checked(b.projection()==['write'],fn)
    got=Body(dump,'task',['s']).val('_0')
    checked(got[0]=='Option::<&SplitPointTask>::expect' and got[1]==('Option::<SplitPointTask>::as_ref',task_cell),'task getter')
    s=Body(dump,'split',['u','ctx','r'])
    s.call(THREAD+'initialize_split_point',['u',slot,'ctx','r'],'init')
    s.call(THREAD+'idle_loop',['u'],'idle')
    s.call(THREAD+'finalize_split_point',['u',slot],'finalize')
    checked(s.projection()==['init','idle','finalize'],'unbracketed owner scope')
    i=Body(dump,'initialize_split_point',['u','s','ctx','r'])
    i.call(THREAD+'active_split_point',['u'],'capture')
    i.call('SplitPointState::set_owner_thread_idx',[('SplitPoint::state','s'),field('u',3)])
    i.call('SplitPoint::set_task',['s',saved],'publish')
    writes(i,writable,('Option::<Arc<SplitPoint>>::Some','s'),'assign')
    i.call('Atomic::<usize>::fetch_add',[field('u',15),'const 1_usize',RELEASE],'up')
    checked(i.projection()==['capture','publish','assign','up'],'owner entry')
    f=Body(dump,'finalize_split_point',['u','s'])
    f.call('Atomic::<bool>::store',[field('u',19),'const true',RELEASE],'wake')
    f.call('Atomic::<usize>::fetch_sub',[field('u',15),'const 1_usize',RELEASE],'down')
    writes(f,writable,field(task,10),'restore')
    f.call('SplitPoint::clear_task',['s'],'clear')
    checked(f.projection()==['wake','down','restore','clear'],'owner return')
    b=Body(dump,'book_into',['u','s','state'])
    writes(b,writable,('Option::<Arc<SplitPoint>>::Some','s'),'book')
    checked(b.projection()==['book'],'booking')
    d=Body(dump,'detach_split',['u','guard'])
    checked(d.val('_0')==('Option::<Arc<SplitPoint>>::take',writable),'taken pointer')
    w=Body(dump,'search_active_split_point',['u'])
    w.call(THREAD+'active_split_point',['u'],'capture')
    w.call(THREAD+'detach_split',['u',(THREAD+'lock','u')],'take')
    w.call('AtomicBitSet::reset',[field(('SplitPoint::state',captured),6),field('u',3)],'reset')
    checked(w.projection()==['capture','take','reset'],'worker bit source')
    x=Body(dump,'abandon_active_split',['u'])
    x.call(THREAD+'detach_split',['u',(THREAD+'lock','u')],'take')
    x.call('AtomicBitSet::reset',[field(('SplitPoint::state',removed),6),field('u',3)],'reset')
    checked(x.paths()=={('take',),('take','reset')},'panic reset target')
    protected=x.call('SplitPoint::lock',[removed])
    owner=x.call('SplitPointState::owner_thread_idx',[('SplitPoint::state',removed)])
    abort=x.call('AbortState::request_abort',[field('u',5)],'abort')
    guard,=edges(x.cfg[owner][-1]);branch=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',x.cfg[guard][-1])
    pred=('Ne',('SplitPointState::owner_thread_idx',('SplitPoint::state',removed)),field('u',3))
    checked(branch and x.val(branch[1])==pred,'foreign-owner panic predicate')
    checked(x.paths(int(branch[2]))=={('reset',)} and x.paths(int(branch[3]))=={('abort','reset')},'foreign panic cancellation')
    checked(Body(dump,'owner_thread_idx',['s']).val('_0')==('Atomic::<usize>::load',field('s',5),RELAXED),'owner load')
    setter=Body(dump,'set_owner_thread_idx',['s','u'])
    setter.call('Atomic::<usize>::store',[field('s',5),'u',RELAXED],'set')
    checked(setter.projection()==['set'],'owner store')
    pc=protected
    while pc!=owner:
        ls=x.cfg[pc];checked(not ls[-1].startswith('drop('),'slot unlocked before owner read')
        pc,=edges(ls[-1])
    return '\n'.join(['import NeuralReversi.PvsParallel.SlotFrameMir.Syntax','', 'namespace '+namespace,'',
        f'def slotsPerThread : Nat := {initbank[2]}',
        'def ownedSlot (u : Worker) (depth : Nat) : Slot := (u,depth)',
        'def savedParent (active : Option Slot) := active',
        'def installed (slot : Slot) := some slot',
        'def restored (taskParent : Option Slot) := taskParent',
        'def workerReset (captured _taken : Option Slot) := captured',
        'def unwindReset (_captured taken : Option Slot) := taken',
        'def unwindAborts (who actualOwner : Worker) : Bool := decide (actualOwner ≠ who)','',
        'end '+namespace,''])


def main():
    run_extractor_cli(
        __doc__,
        render,
        filters=PASS,
        takes_repo=False,
        allow_output=True,
        check_message="slot projection changed",
    )
if __name__=='__main__':main()
