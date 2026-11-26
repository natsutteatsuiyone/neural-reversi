"""Check the actual job option and pointer writes between successive idle dispatches."""
import re
from extract_split_scheduler_mir import Body,edges
from extract_split_activation_mir import THREAD,RELEASE,field,drop_of,idle_authorization
from extract_worker_frame_mir import SUFFIX,TH,inspect_fields
from mir_common import PASS, run_extractor_cli, CALL, checked

def single(b,pattern):
    xs=[(p,re.fullmatch(CALL,ls[-1])) for p,ls in b.cfg.items() if re.search(pattern,ls[-1])]
    checked(len(xs)==1 and xs[0][1],pattern)
    return xs[0]

def render(dump):
    h=Body(dump,'idle_loop',['u'])
    checked(idle_authorization(dump)==(True,True),'empty job and busy acquire no longer guard work')
    take,t=single(h,r'>::take\(')
    worker=h.call(THREAD+'search_active_split_point',['u'],'work')
    helper,run=single(h,r'Thread::run_helper_job\(')
    checked([h.val(arg) for arg in run[3].split(', ')]==['u',('unSome',h.val(t[1]))],
            'helper dispatch receiver or taken job')
    # A take is immediately accounted for by one work or job branch. There is
    # no intervening new scheduler iteration, additional take, or normal exit.
    todo=[(0,False)];seen=set()
    while todo:
        node=todo.pop()
        if node in seen:continue
        seen.add(node);pc,popped=node
        if pc==take:checked(not popped,'an option take was silently discarded');popped=True
        if pc in (helper,worker):checked(popped,'job dispatch without take');popped=False
        if h.cfg[pc][-1]=='return;':checked(not popped,'returned with undispatched job')
        todo.extend((i,popped) for i in edges(h.cfg[pc][-1]))
    b=Body(dump,'book_helper_job',['u','job'])
    loc=b.call(THREAD+'lock',['u'],'lock');guard=re.fullmatch(CALL,b.cfg[loc][-1])[1]
    for pc in drop_of(b,guard):b.marks[pc]=['unlock']
    _,jobLock=single(b,r'>::lock\(')
    checked(b.val(jobLock[3])==field('u',6),'incorrect helper job mutex')
    _,un=single(b,r'>::unwrap\(')
    _,mut=single(b,r'as DerefMut>::deref_mut\(')
    checked(un[3]=='move '+jobLock[1] and b.defs[mut[3].split()[-1]]=={'&mut '+un[1]},'helper mutex guard')
    writes=[(pc,m) for pc,ls in b.cfg.items() for line in ls if (m:=re.fullmatch(r'\(\*(_\d+)\) = move (_\d+);',line))]
    (pc,m),=writes
    checked(m[1]==mut[1],'different job write target')
    definition,=b.defs[m[2]];some=re.fullmatch(r'Option::<.+>::Some\(move (_\d+)\)',definition)
    checked(some,'not a Some write')
    coercion=re.fullmatch(r'move (_\d+) as .+ \(PointerCoercion\(Unsize, Implicit\)\)',next(iter(b.defs[some[1]])))
    checked(coercion and b.val(coercion[1])=='job','incorrect helper job')
    b.marks[pc]=['some'];b.call('Atomic::<bool>::store',[field('u',19),'const true',RELEASE],'wake')
    checked(b.paths()=={('lock','unlock'),('lock','some','wake','unlock')},'helper post order')
    w=Body(dump,'search_active_split_point',['u'])
    w.call(THREAD+'detach_split',['u',(THREAD+'lock','u')],'clear')
    w.call(THREAD+'try_late_join',['u'],'late')
    checked(w.projection()==['clear','late'],'uncleared previous worker invocation')
    target=Body(dump,'run_helper_job',['u','job'])
    got,invoke=single(target,r'>::call\(');target.marks[got]=['job']
    _,deref=single(target,r'Fn\(.+as Deref>::deref\(')
    closure,=re.findall(r'_\d+',next(iter(target.defs[invoke[3].split(', ')[0].split()[-1]])))
    checked(closure==deref[1] and target.val(deref[3])=='job','callback receiver')
    packed,=target.defs[invoke[3].split(', ')[1].split()[-1]]
    receiver=re.fullmatch(r'\(move (_\d+),\)',packed)
    checked(receiver and target.val(receiver[1])=='u','callback worker')
    name=(dump/(TH+'run_helper_job'+SUFFIX)).read_text(encoding='utf8')
    finish=re.search(r'(_\d+) = FinishJob::<.+>\(move (_\d+)\)',name)
    checked(finish and target.val(finish[2])=='u','helper completion worker')
    token=finish[1]
    for pc in drop_of(target,token):target.marks[pc]=['finish']
    checked(target.projection()==['job','finish'],'helper callback return')
    fin=Body(dump,'run_helper_job-{impl#0}-drop',['drop'])
    pc,locked=single(fin,r'Thread::lock\(');fin.marks[pc]=['lock']
    resets=[(pc,m) for pc,ls in fin.cfg.items() if (m:=re.fullmatch(CALL,ls[-1])) and
            m[2]=='Atomic::<bool>::store' and ', const false,' in m[3]]
    (pc,f),=resets
    raw=inspect_fields(fin.cfg,'FinishJob::drop')
    checked((pc,'searching','store','const false','Release') in raw,'helper false store')
    fin.marks[pc]=['false']
    for pc in drop_of(fin,locked[1]):fin.marks[pc]=['unlock']
    checked(fin.projection()==['lock','false','unlock'],'helper completion protection')
    users=set()
    for file in dump.glob('*'+SUFFIX):
        text=file.read_text(encoding='utf8')
        if re.search(r'\.6: std::sync::Mutex<std::option::Option<std::sync::Arc<dyn for<',text):users.add(file.name.removesuffix(SUFFIX))
    checked(users=={TH+'idle_loop',TH+'book_helper_job'},'unknown job option mutator')
    return '\n'.join(['import NeuralReversi.PvsParallel.SlotFrameMir.Live','',
      'namespace NeuralReversi.PvsParallel.SlotFrameMir','',
      'def dispatchWork (searching job : Bool) := searching && !job',
      'def completedHelperBusy := false',
      'def postHelperJob := true','',
      'end NeuralReversi.PvsParallel.SlotFrameMir',''])

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters=PASS,
        takes_repo=False,
        allow_output=True,
        check_message="worker dispatch changed",
    )
if __name__=='__main__':main()
