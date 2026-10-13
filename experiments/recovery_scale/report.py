"""Standalone figures and an evidence-grounded report from frozen study artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import OUT,dump
from .actions import ACTIONS
from .collect import load_split


def pct(x):return f'{100*x:.2f}'
def ci(x):return f"{pct(x['mean'])} [{pct(x['low'])}, {pct(x['high'])}]"


def measured_resources(rows,choices=None):
    """Observed token counts; a kept answer incurs no incremental generation."""
    inputs=np.array([[r['actions'][a]['input_tokens'] for a in ACTIONS] for r in rows])
    outputs=np.array([[r['actions'][a]['output_tokens'] for a in ACTIONS] for r in rows])
    result={'n':len(rows),'offline_total_input_tokens':int(inputs.sum()),
            'offline_total_output_tokens':int(outputs.sum()),'actions':{},'policies':{}}
    for j,a in enumerate(ACTIONS):
        result['actions'][a]={'mean_input_tokens':float(inputs[:,j].mean()),
                              'mean_output_tokens':float(outputs[:,j].mean())}
    idx=np.arange(len(rows))
    for name,c in (choices or {}).items():
        ii=inputs[idx,c]*(np.asarray(c)!=0);oo=outputs[idx,c]*(np.asarray(c)!=0)
        result['policies'][name]={'mean_incremental_input_tokens':float(ii.mean()),
            'mean_incremental_output_tokens':float(oo.mean()),
            'mean_total_input_tokens':float((inputs[:,0]+ii).mean()),
            'mean_total_output_tokens':float((outputs[:,0]+oo).mean())}
    return result


def collection_timings(path,key):
    """Retained cumulative progress clocks, allowing a new clock after resume."""
    sessions=[];active={}
    for line in Path(path).read_text().splitlines():
        try:r=json.loads(line)
        except (ValueError,TypeError):continue
        if not isinstance(r,dict) or r.get('model')!=key or 'done' not in r or 'seconds' not in r:continue
        identity=(r['run'],r['split']);last=active.get(identity)
        if last is None or r['seconds']<last['seconds'] or r['done']<=last['last_done']:
            last={'run':r['run'],'split':r['split'],'first_done':r['done'],'last_done':0,'seconds':0.}
            sessions.append(last);active[identity]=last
        last.update(last_done=r['done'],seconds=r['seconds'])
    return {'observed_collection_seconds':sum(s['seconds'] for s in sessions),'sessions':sessions,
        'interpretation':'Sum of observed per-split collection session clocks in retained progress logs; '
        'excludes model/engine startup, fitting, evaluation, audit and post-last-batch work. '
        'These are offline exhaustive-collection durations, not policy latency. '
        'The overwritten seconds_this_run summary is not used as a lifetime total.'}


def run():
    lines=['# Large recovery-selection experiment','',
        'Two frozen backbones;10,000 fresh diagnostic questions per backbone;10 choices and14 generation calls/question during offline collection. '
        '6000 training,1000 validation,2000 matched test,1000 shifted test. Calibration and repeats are separate. '
        'Selectors use one initial execution. F1 and intervals below are in percentage points.','',
        '## Primary result: one extra generation call per question on average','',
        '| Backbone | Keep | Behavioral | Activation | Correctness gate | Activation − behavioral,95%CI |',
        '|---|---:|---:|---:|---:|---:|']
    summary={};passes=[];fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,key in zip(axes,('qwen7b','phi35')):
        root=OUT/key/'prediction_6000';sel=json.loads((root/'selection.json').read_text())
        ev=json.loads((root/'evaluation.json').read_text());audit=json.loads((OUT/key/'audit.json').read_text())
        names=sel['choices']['10'];r=ev['splits']['test'];primary=r['budgets']['10_calls_1'];m=primary['methods']
        lines.append(f"| {key} | {pct(r['actions']['keep']['f1'])} | {pct(m[names['behavioral']]['f1'])} | "
                     f"{pct(m[names['activation']]['f1'])} | {pct(m[names['gate']]['f1'])} | {ci(primary['activation_minus_behavioral'])} |")
        passes.append(primary['activation_minus_behavioral']['low']>0)
        summary[key]={'selection':names,'primary':primary,'cluster':audit['cluster_intervals'],'splits':{}}
        log=OUT/'logs'/('qwen-main-pipeline.log' if key=='qwen7b' else 'phi-main-pipeline.log')
        summary[key]['collection_timing']=collection_timings(log,key)
        summary[key]['measured_resources']={}
        for split in ('train','validation','test','transfer'):
            rows=[r for p in sorted((OUT/key/'main'/split).glob('batch_*.json'))
                  for r in json.loads(p.read_text())['rows']]
            selected=None
            if split in ('test','transfer'):
                allchoices=dict(np.load(root/f'choices_{split}.npz'))
                selected={label:allchoices['10_calls_1_'+names[label]] for label in ('behavioral','activation','gate')}
            summary[key]['measured_resources'][split]=measured_resources(rows,selected)
        for label,color in [('behavioral','#444444'),('activation','#2369bd'),('gate','#c47125')]:
            xs=[];ys=[]
            for b in (.5,1.,2.):
                p=r['budgets'][f'10_calls_{b:g}']['methods'][names[label]]
                xs.append(p['mean_calls']);ys.append(p['f1']*100)
            ax.plot(xs,ys,'o-',color=color,label=label)
        ax.axhline(r['actions']['keep']['f1']*100,color='gray',ls=':',label='keep')
        ax.set(title=key,xlabel='Mean additional generation calls',ylabel='Test answer F1 (%)');ax.grid(alpha=.2)
        for split in ('test','transfer'):
            d=load_split(key,split);y=d['f1'];em=d['em'];error=em[:,0]==0
            response={'initial_errors':int(error.sum()),'any_rescue':int((error & (em[:,1:]==1).any(1)).sum()),
                'no_observed_rescue':int((error & ~(em[:,1:]==1).any(1)).sum()),
                'new_menu_only_rescue':int((error & ~(em[:,1:5]==1).any(1) & (em[:,5:]==1).any(1)).sum()),
                'by_action':{a:{'rescue':int((error & (em[:,j]==1)).sum()),
                    'harm':int((~error & (em[:,j]==0)).sum()),
                    'unique_rescue':int((error & (em[:,j]==1) & (em[:,1:].sum(1)==1)).sum()),
                    'empty_rate':float(np.mean([not row['actions'][a]['answer'] for row in d['rows']])),
                    'at_cap_rate':float(np.mean([row['actions'][a]['at_cap'] for row in d['rows']]))}
                    for j,a in enumerate(ACTIONS[1:],1)}}
            rescued=(em[error,1:]==1).astype(np.int64)
            response['pairwise_rescue_intersections']={a:{b:int((rescued.T@rescued)[i,j])
                for j,b in enumerate(ACTIONS[1:])} for i,a in enumerate(ACTIONS[1:])}
            summary[key]['splits'][split]={'response':response,'outcomes':ev['splits'][split]}
        repeat_report={};main=load_split(key,'test');choices=dict(np.load(root/'choices_test.npz'))
        for repeat in (1,2):
            repeated=load_split(key,'test',run_name=f'repeat{repeat}');n=len(repeated['rows'])
            rr={'n':n,'actions':{},'fixed_original_policy_f1':{},'main_subset_policy_f1':{}}
            for j,a in enumerate(ACTIONS):
                rr['actions'][a]={'answer_agreement':float(np.mean([x['actions'][a]['answer']==z['actions'][a]['answer']
                      for x,z in zip(main['rows'][:n],repeated['rows'])])),
                      'f1':float(repeated['f1'][:,j].mean()),'main_subset_f1':float(main['f1'][:n,j].mean())}
            for label in ('behavioral','activation','gate'):
                c=choices['10_calls_1_'+names[label]][:n]
                rr['fixed_original_policy_f1'][label]=float(repeated['f1'][np.arange(n),c].mean())
                rr['main_subset_policy_f1'][label]=float(main['f1'][np.arange(n),c].mean())
            delta=repeated['states'].astype(np.float32)-main['states'][:n].astype(np.float32)
            rr['state_relative_l2_difference']=float(np.linalg.norm(delta.ravel())/np.linalg.norm(main['states'][:n].astype(np.float32).ravel()))
            repeat_report[str(repeat)]=rr
        summary[key]['repeats']=repeat_report
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'policy_frontier.png',dpi=180);fig.savefig(OUT/'policy_frontier.pdf');plt.close(fig)
    lines+=['',f"Primary criterion (positive question-bootstrap lower bound on both backbones): **{'met' if all(passes) else 'not met'}**.",'',
        'The criterion concerns the tested feature families,action menu,resource budget and task. It does not establish a causal faulty layer or universal recoverability.','',
        '![Policy frontier](policy_frontier.png)','', '## Shift, controls and action diversity','']
    for key in ('qwen7b','phi35'):
        s=summary[key];ev=s['splits'];lines += [f'### {key}','',
            f"Selected behavioral: `{s['selection']['behavioral']}`; activation: `{s['selection']['activation']}`; correctness gate: `{s['selection']['gate']}`.",'',
            f"Main test activation−gate: {ci(s['primary']['activation_minus_gate'])}.",
            f"Main test article-component interval for activation−behavioral: {ci(s['cluster']['test']['activation_minus_behavioral'])}; "
            f"{s['cluster']['test']['activation_minus_behavioral']['groups']} components.",
            f"Shift activation−behavioral: {ci(ev['transfer']['outcomes']['budgets']['10_calls_1']['activation_minus_behavioral'])}.",
            f"Expanded-menu−old-menu activation selector: {ci(ev['test']['outcomes']['expanded_menu_minus_old_activation'])}.",
            f"Expanded-menu−old-menu behavioral selector: {ci(ev['test']['outcomes']['expanded_menu_minus_old_behavioral'])}.",'',
            'Primary policy outcomes:','',
            '| Policy | F1 | Exact match | Initial-error rescue rate | Initially-correct harm rate | Mean calls |',
            '|---|---:|---:|---:|---:|---:|']
        for label in ('behavioral','activation','gate'):
            v=s['primary']['methods'][s['selection'][label]]
            lines.append(f"| {label} | {pct(v['f1'])} | {pct(v['em'])} | {pct(v['rescue_rate'])} | {pct(v['harm_rate'])} | {v['mean_calls']:.3f} |")
        lines+=['',f"Activation−behavioral exact-match difference: {ci(s['primary']['activation_minus_behavioral_em'])}.",
            f"Hindsight one-call-budget oracle: feasible F1 {pct(s['primary']['oracle']['f1'])}, "
            f"dual upper bound {pct(s['primary']['oracle']['dual_f1_upper'])}. The oracle uses gold outcomes and is not deployable.",'',
            'Primary action counts:','',
            '| Policy | '+ ' | '.join(ACTIONS)+' |','|---|'+'---:|'*len(ACTIONS)]
        for label in ('behavioral','activation','gate'):
            counts=s['primary']['methods'][s['selection'][label]]['action_counts']
            lines.append('| '+label+' | '+' | '.join(str(v) for v in counts)+' |')
        lines+=['','Single-question decisions use the multiplier selected on validation,without reallocating a new batch. '
            'Their actual mean calls can exceed the validation target after distribution shift.','',
            '| Split | Online policy | F1 | Exact match | Actual mean calls |','|---|---|---:|---:|---:|']
        for split in ('test','transfer'):
            for label in ('behavioral','activation','gate'):
                v=ev[split]['outcomes']['online']['10_'+label]
                lines.append(f"| {split} | {label} | {pct(v['f1'])} | {pct(v['em'])} | {v['mean_calls']:.3f} |")
        lines+=['',
            '| Action | All-action test F1 | Error rescues | Initially correct answers harmed | Unique rescues |',
            '|---|---:|---:|---:|---:|']
        response=ev['test']['response']
        for a in ACTIONS[1:]:
            v=response['by_action'][a];lines.append(f"| {a} | {pct(ev['test']['outcomes']['actions'][a]['f1'])} | {v['rescue']} | {v['harm']} | {v['unique_rescue']} |")
        lines+=['','| Action | No parsed answer (%) | Any stage at cap (%) |','|---|---:|---:|']
        for a,v in response['by_action'].items():
            lines.append(f"| {a} | {pct(v['empty_rate'])} | {pct(v['at_cap_rate'])} |")
        lines+=['',f"Of {response['initial_errors']} initial errors,{response['any_rescue']} have at least one observed rescue; "
            f"{response['new_menu_only_rescue']} are rescued only by an added action;{response['no_observed_rescue']} have no rescue in this menu. "
            'Action rescue categories overlap. These labels describe recorded responses,not internal causal mechanisms.','',
            'Learning curves (same held-out questions; independently fitted training-prefix encoders/selectors):','',
            '| Training questions | Behavioral F1 | Activation F1 | Difference,95%CI |','|---|---:|---:|---:|']
        curves={}
        for size in (1200,3000,6000):
            root=OUT/key/f'prediction_{size}';ss=json.loads((root/'selection.json').read_text())['choices']['10']
            ee=json.loads((root/'evaluation.json').read_text())['splits']['test']['budgets']['10_calls_1']
            lines.append(f"| {size} | {pct(ee['methods'][ss['behavioral']]['f1'])} | {pct(ee['methods'][ss['activation']]['f1'])} | {ci(ee['activation_minus_behavioral'])} |")
            curves[str(size)]=ee
        s['learning_curves']=curves
        lines+=['','Repeated execution on the first 128 test questions,with the original policy choices held fixed:','',
            '| Policy | Main subset F1 | Repeat 1 F1 | Repeat 2 F1 |','|---|---:|---:|---:|']
        for label in ('behavioral','activation','gate'):
            r1,r2=s['repeats']['1'],s['repeats']['2']
            lines.append(f"| {label} | {pct(r1['main_subset_policy_f1'][label])} | {pct(r1['fixed_original_policy_f1'][label])} | {pct(r2['fixed_original_policy_f1'][label])} |")
        lines+=['',f"Baseline-state relative L2 difference was {r1['state_relative_l2_difference']:.3g} and "
            f"{r2['state_relative_l2_difference']:.3g}. Recovery answer strings can change,including in greedy branches; "
            'per-action agreement and F1 are recorded in additional_analysis.json. These repeats characterize execution sensitivity on a subset.','']
        sel=json.loads((OUT/key/'prediction_6000'/'selection.json').read_text())
        lines+=['','Feature ablations (each response model selected on validation; secondary comparisons):','',
            '`random` and `full_random` use fixed random projections of hidden states.','',
            '| Feature family | Selected predictor | Test F1 | Test exact match |','|---|---|---:|---:|']
        for family,name in sel['choices']['10']['family_best'].items():
            v=s['primary']['methods'][name]
            lines.append(f"| {family} | `{name}` | {pct(v['f1'])} | {pct(v['em'])} |")
        lines+=['','Conservative token-reservation budgets (multiples of validation mean both-action reservation):','',
            '| Budget multiple | Behavioral F1 | Activation F1 | Activation−behavioral,95%CI |','|---|---:|---:|---:|']
        for fraction in (.5,1.,2.):
            v=ev['test']['outcomes']['budgets'][f'10_tokens_{fraction:g}']
            lines.append(f"| {fraction:g} | {pct(v['methods'][s['selection']['behavioral']]['f1'])} | "
                         f"{pct(v['methods'][s['selection']['activation']]['f1'])} | {ci(v['activation_minus_behavioral'])} |")
        resources=s['measured_resources'];tot_in=sum(v['offline_total_input_tokens'] for v in resources.values())
        tot_out=sum(v['offline_total_output_tokens'] for v in resources.values())
        elapsed=s['collection_timing']['observed_collection_seconds']
        lines+=['','Measured resources (token counts sum every call within an action):','',
            '| Test action | Mean input tokens | Mean output tokens |','|---|---:|---:|']
        for a,v in resources['test']['actions'].items():
            lines.append(f"| {a} | {v['mean_input_tokens']:.1f} | {v['mean_output_tokens']:.1f} |")
        lines+=['','| One-call-budget test policy | Additional input tokens | Additional output tokens | Total input tokens including baseline | Total output tokens including baseline |',
            '|---|---:|---:|---:|---:|']
        for label,v in resources['test']['policies'].items():
            lines.append(f"| {label} | {v['mean_incremental_input_tokens']:.1f} | {v['mean_incremental_output_tokens']:.1f} | "
                         f"{v['mean_total_input_tokens']:.1f} | {v['mean_total_output_tokens']:.1f} |")
        lines+=['',f'Main exhaustive collection: {tot_in:,} input tokens and {tot_out:,} output tokens. '
            f'Observed collection segment time including repeats: {elapsed/3600:.2f} hours.',
            s['collection_timing']['interpretation'],
            'Inputs are request token counts; recorded output lengths exclude terminal EOS/padding. '
            'Prefix caching can reduce physical prefill work.','']
    lines += ['', '## Reproducibility and limits','',
        '- Full raw audits reconstruct official scores,prompts,stage costs,metadata,and all held-out choices through reloaded raw-state encoders and predictors.',
        '- Baseline: Transformers BF16/SDPA. Recovery: vLLM0.22 BF16. Exact cross-engine or repeated-execution equality is not assumed;128 questions were repeated twice per backbone. Detailed agreement and fixed-policy repeat scores are in additional_analysis.json.',
        '- Primary budget counts generation calls. Token-reservation analyses use separate conservative caps; neither measures FLOPs or end-to-end latency. Baseline capture and selector overhead are outside incremental action budgets.',
        '- Question bootstrap is conditional on fitted models,selected policies,fixed evaluated batch choices and recorded action executions. Article-component intervals address one source of question dependence; they do not include training-sample or full execution uncertainty.',
        '- Hybrid collection keeps separate Transformers/vLLM copies of each backbone. Its calibration speedup describes offline collection throughput; end-to-end routing latency and memory efficiency were not benchmarked.',
        '- Primary difficulty/type matching removes the pilot’s mixed-to-hard mismatch. Topic-title embargo and the transfer test create other shifts. The transfer test changes both source difficulty and initial evidence count.',
        '- HotpotQA distractor pools already contain relevant evidence. Results do not demonstrate open-web retrieval or generalization to other task domains. Two backbones are separate fitted replications.',
        '- The expanded menu contains operational recovery strategies. A failed tested menu does not imply that an answer cannot be fixed. No causal-layer localization claim is made.',
        '- Regression,compression,and budget allocation are standard components. Broad activation-based retrieval/strategy selection overlaps prior work including [SeaKR](https://aclanthology.org/2025.acl-long.1312/). No algorithmic novelty claim is established here.','']
    dump(OUT/'additional_analysis.json',summary);(OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'primary_criterion_met':all(passes),'report':str(OUT/'REPORT.md')}))


if __name__=='__main__':run()
