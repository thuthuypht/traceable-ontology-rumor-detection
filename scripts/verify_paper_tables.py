"""Verify selected reported numbers from frozen aggregate CSVs; no new inference."""
from pathlib import Path
import csv
import sys
P=Path(__file__).resolve().parent.parent

def read(rel):
    with (P/rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(observed,target,tol=1e-6):
    assert abs(float(observed)-target)<tol,(observed,target)
rows={r['model']:r for r in read('results/twitter_covid19/external_summary_metrics.csv')}
close(rows['C_shared']['macro_f1'],0.474390595245624)
close(rows['D_shared']['macro_f1'],0.4604586101813459)
close(rows['D_shared']['mcc'],-0.07019515600066104)
close(rows['D_shared']['auprc'],0.6892583535247312)
close(rows['C_shared']['auprc'],0.6744016270806592)
assert int(rows['D_shared']['tn'])+int(rows['D_shared']['fp'])+int(rows['D_shared']['fn'])+int(rows['D_shared']['tp'])==400
r=read('results/ontology216_faithfulness/reports/rule_faithfulness_summary.csv')
assert len(r)==6 and all(int(x['n_case_rule_pairs'])==216 for x in r)
assert sum(round(float(x['fraction_flipped'])*216) for x in r)==29
m=read('results/ontology216_faithfulness/reports/historical_prediction_reconstruction_audit.csv')
assert len(m)==8 and all(x['exact_reference_match_at_1e-4']=='True' for x in m)
assert max(float(x['max_abs_p_diff_D_vs_historical']) for x in m)<2e-15
s=read('results/pheme_v16/summary_by_config.csv')
ss={x['Config']:x for x in s}
close(ss['C_Graph_RawEvidence']['MacroF1_mean'],0.7090230829485156)
close(ss['H_Ontology_ReliabilityGate']['MacroF1_mean'],0.7105496918009293)
print('PASS: exact numeric assertions, n=400 external, 216 cases x 6 rules, 8 model configurations, PHEME v16.')
