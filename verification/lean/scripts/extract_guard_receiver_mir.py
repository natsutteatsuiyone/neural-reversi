"""Extract the incoming scheduler calls and the actual guarded recursive split receiver."""
import re
from extract_split_activation_mir import THREAD, field, idle_authorization
from extract_split_scheduler_mir import Body, edges
from extract_split_rebooking_mir import nested
from extract_worker_frame_mir import KINDS, PL, SUFFIX, TH, ANY_FIELD, inspect_fields
from mir_common import PASS, run_extractor_cli, blocks, CALL, checked

PVS='reversi_core.search-pvs-search'
SCOUT=PVS+'_split_point'
DROP=TH.replace('{impl#1}-','{impl#0}-')+'drop'
TYPES={**KINDS,TH+'dispatch_search':'dispatch',TH+'initialize_split_point':'ownerInit',
       PL+'assign_helpers_to_split_point':'assign',TH+'try_late_join':'late',
       TH+'search_active_split_point-{impl#0}-drop':'unwindWork',DROP:'unwindSplit',
       PVS:'pvs',SCOUT:'scout'}
METHODS={TH+n:n for n in ('dispatch_search','initialize_split_point','finalize_split_point','try_late_join',
          'book_into','search_active_split_point','run_helper_job','abandon_active_split','detach_split',
          'split','main_thread_loop','idle_loop')}
METHODS[PL+'assign_helpers_to_split_point']='assign_helpers_to_split_point'

def symbolic(path,roots):
    body=object.__new__(Body)
    body.cfg=blocks(path.read_text(encoding='utf8'));body.defs={};body.marks={};body.roots=roots
    for b in body.cfg.values():
        for line in b:
            if m:=re.fullmatch(r'(_\d+) = (.+?)(?: -> .+)?;',line):
                body.defs.setdefault(m[1],set()).add(m[2])
    return body

def idle_store_receiver(dump):
    for name,roots,expected in [('detach_split',['worker','guard'],field('worker',19)),
            ('main_thread_loop',['worker','channel'],field('worker',19)),
            ('run_helper_job-{impl#0}-drop',['guard'],
             '((*((*_1).0: &search::threading::thread::Thread)).19: util::align::Align64<std::sync::atomic::Atomic<bool>>)')]:
        b=Body(dump,name,roots)
        found=[pc for pc,field_name,op,value,_ in inspect_fields(b.cfg,name)
               if (field_name,op,value)==('searching','store','const false')]
        site,=found
        call=re.fullmatch(CALL,b.cfg[site][-1])
        checked(b.val(call[3].split(', ')[0])==expected,'idle flag written to another thread')
    ctor=Body(dump,'initialize_split_point',['worker','slot','ctx','req'])
    text=(dump/(TH+'initialize_split_point'+SUFFIX)).read_text(encoding='utf8')
    fields=re.search(r" = SplitUnwindGuard::<'_> \{ thread: move (_\d+), sp: move (_\d+) \}",text)
    checked(fields and [ctor.val(fields[i]) for i in (1,2)]==['worker','slot'],'changed unwind slot or owner')
    for name,guard in [('search_active_split_point','UnwindCleanup'),('run_helper_job','FinishJob')]:
        b=Body(dump,name,['worker','job'] if name=='run_helper_job' else ['worker'])
        text=(dump/(TH+name+SUFFIX)).read_text(encoding='utf8')
        saved=re.search(r" = "+guard+r"::<'_>\(move (_\d+)\)",text)
        checked(saved and b.val(saved[1])=='worker','changed cleanup receiver')
    for name in (DROP,TH+'search_active_split_point-{impl#0}-drop'):
        b=symbolic(dump/(name+SUFFIX),{1:'guard'})
        b.call(THREAD+'abandon_active_split',['(*((*_1).0: &search::threading::thread::Thread))'])
    abandon=Body(dump,'abandon_active_split',['worker'])
    abandon.call(THREAD+'detach_split',['worker',(THREAD+'lock','worker')])

def pvs_guard(dump):
    files=[dump/(k+SUFFIX) for k in (PVS,SCOUT)]
    p=symbolic(files[0],{6:'worker'})
    guard=p.call(THREAD+'can_split',['worker'])
    splits=[(pc,m) for pc,ls in p.cfg.items() if THREAD+'split(' in ls[-1] and (m:=re.fullmatch(CALL,ls[-1]))]
    (split,call),=splits
    checked(p.val(call[3].split(', ')[0])=='worker','split receiver differs from guard')
    g=re.fullmatch(CALL,p.cfg[guard][-1])
    branch=re.fullmatch(r'switchInt\(move '+g[1]+r'\) -> \[0: bb(\d+), otherwise: bb(\d+)\];',p.cfg[int(g[4])][-1])
    checked(branch and int(branch[2])==split,'split is not immediate guarded branch')
    for pc,ls in p.cfg.items():
        if pc==int(g[4]):continue
        checked(not re.search(r'\bbb'+str(split)+r'\b',ls[-1]),'bypassed can_split')
    for body in (p,symbolic(files[1],{4:'worker'})):
        sites=[m for ls in body.cfg.values() if ' = search::<' in ls[-1] and (m:=re.fullmatch(CALL,ls[-1]))]
        checked(sites and all(body.val(m[3].split(', ')[5])=='worker' for m in sites),'recursive child changed worker')
    w=Body(dump,'search_active_split_point',['worker'])
    active=('unSome',(THREAD+'active_split_point','worker'))
    dispatch=[(pc,m) for pc,ls in w.cfg.items() if THREAD+'dispatch_search(' in ls[-1]
              and (m:=re.fullmatch(CALL,ls[-1]))]
    (dispatch_site,hit),=dispatch
    args=[w.val(x) for i,x in enumerate(hit[3].split(', ')) if i in (0,2,3,4)]
    checked(args==['worker',('SplitPoint::task',active),('SplitPoint::move_iter',active),active],'captured dispatch operands')
    w.marks[dispatch_site]=['dispatch'];w.call(THREAD+'active_split_point',['worker'],'capture')
    w.call(THREAD+'detach_split',['worker',(THREAD+'lock','worker')],'detach')
    checked(w.projection()==['capture','dispatch','detach'],'worker frame interval')
    dispatch=Body(dump,'dispatch_search',['worker','ctx','task','moves','slot'])
    nodes=[m for ls in dispatch.cfg.values() if ' = search_split_point::<' in ls[-1] and (m:=re.fullmatch(CALL,ls[-1]))]
    checked(len(nodes)==3 and all([dispatch.val(x) for x in m[3].split(', ')]==
            ['ctx','task','moves','worker','slot'] for m in nodes),'worker dispatch forwarding')
    checked(idle_authorization(dump)==(True,True),'idle work entry conditions')
    nested(dump,8)
    # Isolate the particular busy-guard success which leads to this work
    # invocation. A previous helper callback is in an earlier scheduler turn.
    h=Body(dump,'idle_loop',['worker'])
    guarded=[
        (i,m) for i,ls in h.cfg.items() if (m:=re.fullmatch(CALL,ls[-1])) and m[2]=='Atomic::<bool>::load' and
        [h.val(v) for v in m[3].split(', ')]==[field('worker',19),'std::sync::atomic::Ordering::Acquire']]
    (busy,check),=guarded
    choose=int(check[4])
    term=re.fullmatch(r'switchInt\(move '+check[1]+r'\) -> \[0: bb\d+, otherwise: bb(\d+)\];',h.cfg[choose][-1])
    checked(term,'unrecognized busy-guard launch')
    helper=[pc for pc,ls in h.cfg.items() if THREAD+'run_helper_job(' in ls[-1]]
    start=h.call(THREAD+'search_active_split_point',['worker'])
    todo=[int(term[1])];slots=set()
    while todo:
        i=todo.pop()
        if i in slots or i in (busy,start,*helper):continue
        slots.add(i);todo.extend(edges(h.cfg[i][-1]))
    allowed=(' as Deref>::deref','>::lock','>::unwrap',' as DerefMut>::deref_mut','>::take')
    for i in slots:
        line=h.cfg[i][-1]
        if m:=re.fullmatch(CALL,line):checked(m[2].endswith(allowed),'unexpected launch code '+m[2])
    # All calls on the work prologue before dispatch operate on the locked
    # active clone, its task and move iterator, and the derived child context.
    todo=[0];seen=set()
    while todo:
        i=todo.pop()
        if i in seen or i==dispatch_site:continue
        seen.add(i);line=w.cfg[i][-1]
        if m:=re.fullmatch(CALL,line):
            checked(m[2] in {THREAD+'lock',THREAD+'active_split_point','SplitPoint::task','SplitPoint::move_iter',
                        "SearchContext::<'_>::from_split_task"} or
                    m[2].endswith((' as Deref>::deref',' as Clone>::clone','>::expect')),'changed work prologue')
        todo.extend(edges(line))

def render(dump):
    pvs_guard(dump)
    idle_store_receiver(dump)
    incoming={TYPES[k]:set() for k in METHODS}
    incoming.update({k:set() for k in ('unwindSplit','unwindWork','helperDrop','mainEntry','workerEntry')})
    matcher=re.compile(r'\b(?:Thread|ThreadPool)::('+'|'.join(METHODS.values())+r')\b')
    byname={v:TYPES[k] for k,v in METHODS.items()}
    guardUsers={g:set() for g in ('SplitUnwindGuard','UnwindCleanup','FinishJob')}
    active=set(); flags=set()
    for f in dump.glob('*'+SUFFIX):
        name=f.name.removesuffix(SUFFIX);role=TYPES.get(name,'other')
        raw=f.read_text(encoding='utf8')
        checked('catch_unwind' not in raw,'panic can resume inside crate body '+name)
        if role in ('work','ownerInit','split','pvs','scout','dispatch'):
            clean=re.findall(r'^    bb(\d+) \(cleanup\): \{\n((?:        [^\n]+\n)+)    \}',raw,re.M)
            cleanup={i for i,_ in clean}
            for _,body in clean:
                checked(set(re.findall(r'\bbb(\d+)\b',body.splitlines()[-1]))<=cleanup,'cleanup returns to normal code')
        for guard,names in guardUsers.items():
            if guard in raw:names.add(role)
        for m in matcher.finditer(raw):
            line=raw[raw.rfind('\n',0,m.start())+1:raw.find('\n',m.end())]
            if not line.startswith('        '):continue
            checked(m[0]+'(' in line and ' -> [' in line,'escaped scheduler method '+line)
            incoming[byname[m[1]]].add(role)
        if ANY_FIELD.search(raw):
            for _,fld,op,v,order in inspect_fields(blocks(raw),name):
                if fld=='searching' and op=='store':
                    checked(order=='Release','searching write ordering');flags.add((role,v.split()[-1]))
        if THREAD+'active_split_point_mut(' in raw:active.add(role)
    checked(guardUsers=={'SplitUnwindGuard':{'unwindSplit','ownerInit','split'},
                         'UnwindCleanup':{'unwindWork','work'},'FinishJob':{'helperDrop','helper'}},
            'new holder of scheduler destructor: '+str(guardUsers))
    for k,owners in [('unwindSplit',['ownerInit','split']),('unwindWork',['work']),('helperDrop',['helper'])]:
        incoming[k].update(owners)
    checked(incoming['split']=={'pvs'} and incoming['dispatch']=={'work'},'different guarded split origins')
    checked(active=={'ownerInit','finalize','book','detach'},'active Arc writer inventory: '+str(active))
    checked(flags and 'other' not in {k for k,_ in flags},'unknown searching write')
    r=lambda xs: ', '.join('.'+x for x in sorted(xs))
    output=['import NeuralReversi.PvsParallel.SlotFrameMir.Syntax','',
            'namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver','',
            'inductive Site','  | '+' | '.join(sorted({'other',*TYPES.values()})),
            '  deriving DecidableEq','',
            'def incoming : Site → Option (List Site)']
    output += [f'  | .{k} => some [{r(v)}]' for k,v in sorted(incoming.items())]
    output += ['  | _ => none','',f'def pointerWriters : List Site := [{r(active)}]','',
               'def searchingWriters : List (Site × Bool) := ['+', '.join(f'(.{k}, {b})' for k,b in sorted(flags))+']','',
               'def workKind : Site → WorkerFrameMir.Kind']
    output += [f'  | .{site} => .{site}' for site in sorted(set(KINDS.values()))]
    output += ['  | _ => .other','','def guardReceiver (active : Option Slot) := active','',
               'end NeuralReversi.PvsParallel.SlotFrameMir.Receiver','']
    return '\n'.join(output)

def main():
    run_extractor_cli(
        __doc__,
        render,
        filters=PASS,
        takes_repo=False,
        allow_output=True,
        check_message="guard receiver changed",
    )
if __name__=='__main__':main()
