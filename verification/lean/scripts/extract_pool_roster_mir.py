"""Identify which thread books in each phase and the unique pool cursor used by startup."""
import re

from extract_split_activation_mir import THREAD,field
from extract_split_scheduler_mir import Body,edges
from extract_worker_frame_mir import SUFFIX,TH,PL
from mir_common import PASS, run_extractor_cli, CALL, checked


def cursor(body,initialize,advance,consume,must_consume):
    # Each successful consume is preceded by exactly one Some result from the
    # collection's single Iterator::next. No branch can consume it twice.
    out=re.fullmatch(CALL,body.cfg[advance][-1])[1]
    branch,=edges(body.cfg[advance][-1]);end=body.cfg[branch][-1]
    m=re.fullmatch(r'switchInt\(move (_\d+)\) -> \[0: bb(\d+), 1: bb(\d+), otherwise: bb(\d+)\];',end)
    checked(m and body.defs[m[1]]=={'discriminant('+out+')'},'unexpected iterator pattern')
    todo=[(0,False,False)];seen=set();visited=False
    while todo:
        state=todo.pop()
        if state in seen:continue
        seen.add(state);pc,live,exists=state
        if pc==initialize:
            checked(not exists,'iterator restarted');exists=True
        if pc==advance:
            checked(exists and not (live and must_consume),'missing worker registration')
            live=False
        if body.cfg[pc][-1]=='return;':checked(not (live and must_consume),'skipped final registration')
        if pc==consume:
            checked(live,'cursor consumed twice or before next')
            live=False;visited=True
        if pc==branch:
            todo.extend([(int(m[2]),False,exists),(int(m[3]),True,exists)])
        else:
            todo.extend((j,live,exists) for j in edges(body.cfg[pc][-1]))
    checked(visited,'unreachable cursor consumption')


def render(dump):
    # Only the two protected implementations can ever book a split point.
    sites=set();writers=set()
    for f in dump.glob('*'+SUFFIX):
        src=f.read_text(encoding='utf8')
        for line in src.splitlines():
            if '.0: std::vec::Vec<std::sync::Arc<search::threading::thread::Thread>>' in line:
                if not re.fullmatch(r'\s*_\d+ = &\(\(\*_\d+\)\.0: .+\);',line):
                    checked(re.fullmatch(r'\s*_\d+ = &mut \(\(\*_\d+\)\.0: .+\);',line),'unsupported roster field access')
                    writers.add(f.name.removesuffix(SUFFIX))
            if line.startswith('        ') and THREAD+'book_into' in line:
                m=re.fullmatch(CALL,line.strip())
                checked(m and m[2]==THREAD+'book_into','indirect split reservation')
                sites.add(f.name.removesuffix(SUFFIX))
    checked(sites=={PL+'assign_helpers_to_split_point',TH+'try_late_join'},'new split enrollment site')
    checked(writers=={PL+'add_thread',PL+'exit'},'new pool mutation')
    ctor=(dump/(PL+'new-{closure#0}'+SUFFIX)).read_text(encoding='utf8')
    init=re.search(r'ThreadPool \{ threads: move (_\d+),',ctor)
    checked(init and re.search(re.escape(init[1])+r' = Vec::<Arc<threading::thread::Thread>>::new\(\)',ctor),'nonempty initial pool')
    b=Body(dump,'pool-{impl#0}-init',['pool','weak','rx'])
    b.call('ThreadPool::create_main_thread',['pool','weak','rx'],'main')
    b.call('ThreadPool::create_worker_threads',['pool','weak'],'workers')
    checked(b.projection()==['main','workers'],'constructor order')
    b=Body(dump,'create_main_thread',['pool','weak','rx'])
    b.call('ThreadPool::add_thread',['pool','const 0_usize','weak'],'zero')
    checked(b.projection()==['zero'],'primary ID')
    b=Body(dump,'create_worker_threads',['pool','weak'])
    into='<std::ops::Range<usize> as IntoIterator>::into_iter';nxt='<std::ops::Range<usize> as Iterator>::next'
    r=next(x for options in b.defs.values() for x in options if x.startswith('std::ops::Range::<usize> {'))
    m=re.fullmatch(r'std::ops::Range::<usize> \{ start: const (\d+)_usize, end: move (_\d+) \}',r)
    checked(m and m[1]=='1' and b.val(m[2])==field('pool',2),'helper range')
    i=b.call(nxt,[(into,r)])
    j=b.call('ThreadPool::add_thread',['pool',('unSome',(nxt,(into,r))),'weak'])
    cursor(b,b.call(into,[r]),i,j,True)
    b=Body(dump,'add_thread',['pool','idx','weak'])
    made=(THREAD+'new','idx',field('pool',3),field('pool',5),field('pool',6),field('pool',2),field('pool',7),'weak')
    b.call('Vec::<Arc<threading::thread::Thread>>::push',[field('pool',0),('Arc::<threading::thread::Thread>::new',made)],'push')
    checked(b.projection()==['push'],'pool entry push')
    thread=(dump/(TH+'new'+SUFFIX)).read_text(encoding='utf8')
    m=re.search(r'threading::thread::Thread \{ mutex_for_sleep_condition: [^\n]+?idx: (?:copy|move) (_\d+),',thread)
    checked(m and (m[1]=='_1' or re.search(re.escape(m[1])+r' = (?:copy|move) _1;',thread)),'thread ID')
    b=Body(dump,'assign_helpers_to_split_point',['pool','slot'])
    into='<&Vec<Arc<threading::thread::Thread>> as IntoIterator>::into_iter'
    nxt="<std::slice::Iter<'_, Arc<threading::thread::Thread>> as Iterator>::next"
    owner=(into,field('pool',0))
    i=b.call(nxt,[owner])
    j=b.call(THREAD+'book_into',[('unSome',(nxt,owner)),'slot',('SplitPoint::state','slot')])
    cursor(b,b.call(into,[field('pool',0)]),i,j,False)
    late=Body(dump,'try_late_join',['u'])
    m,=[m for ls in late.cfg.values() if (m:=re.fullmatch(CALL,ls[-1])) and m[2]==THREAD+'book_into']
    checked(late.val(m[3].split(', ')[0])=='u','late reservation is not by the target thread')
    return '\n'.join(['import NeuralReversi.PvsParallel.SlotFrameMir.Syntax','',
        'namespace NeuralReversi.PvsParallel.SlotFrameMir','',
        "def poolIndices (size : Nat) : List Nat := 0 :: List.range' 1 (size-1)",
        'def lateTarget (worker : Worker) : Worker := worker','',
        'end NeuralReversi.PvsParallel.SlotFrameMir',''])

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters=PASS,
        takes_repo=False,
        allow_output=True,
        check_message="pool roster changed",
    )

if __name__=='__main__':main()
