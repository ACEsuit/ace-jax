import json,glob,sys
d=sys.argv[1] if len(sys.argv)>1 else 'runs/density_qoi'
ref=json.load(open(f'{d}/mace_sige.json'))
cols=[(ph,q) for ph in ("Si","Ge","SiGe") for q in ("a0_cubic","B_GPa","C11_GPa","C12_GPa","C44_GPa")]
print(f"{'SiGe model':42s} {'elastic |%|':>11s} {'Si vac dE':>10s} {'Ge vac dE':>10s}   (MACE Si {ref['Si']['vacancy_eV']:.2f}, Ge {ref['Ge']['vacancy_eV']:.2f} eV)")
for f in sorted(glob.glob(f'{d}/ace_sige_*.json')):
    m=json.load(open(f)); dev=[abs(100*(m[p][q]-ref[p][q])/abs(ref[p][q])) for p,q in cols]
    print(f"{f.split('ace_')[1][:-5]:42s} {sum(dev)/len(dev):11.2f} {m['Si']['vacancy_eV']-ref['Si']['vacancy_eV']:+10.3f} {m['Ge']['vacancy_eV']-ref['Ge']['vacancy_eV']:+10.3f}")
rc=json.load(open(f'{d}/mace_cantor.json'))
def der(x): return {"B":x["B_GPa"],"C44":x["C44_GPa"]}
rv=rc["draws"][0]["vacancy_eV"]
print(f"\n{'Cantor model':42s} {'B % per draw':>22s} {'C44 % per draw':>22s} {'vac MAE eV':>11s}")
for f in sorted(glob.glob(f'{d}/ace_cantor_*.json')):
    m=json.load(open(f))
    B=[100*(a["B_GPa"]-b["B_GPa"])/b["B_GPa"] for a,b in zip(m["draws"],rc["draws"])]
    C=[100*(a["C44_GPa"]-b["C44_GPa"])/b["C44_GPa"] for a,b in zip(m["draws"],rc["draws"])]
    mv=m["draws"][0]["vacancy_eV"]; e=[abs(x-y) for el in rv for x,y in zip(mv[el],rv[el])]
    print(f"{f.split('ace_')[1][:-5]:42s} {' / '.join(f'{x:+.0f}' for x in B):>22s} {' / '.join(f'{x:+.0f}' for x in C):>22s} {sum(e)/len(e):11.3f}")
