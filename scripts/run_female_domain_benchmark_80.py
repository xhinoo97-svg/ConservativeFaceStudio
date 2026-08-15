from __future__ import annotations

from pathlib import Path
import sys
import urllib.request

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Apply packaged process defaults.
from app import female_domain_benchmark as benchmark
from scripts import run_female_domain_benchmark_observed as observed
from scripts.run_female_domain_benchmark_resilient import _resilient_urlopen


# Public identity/biographical metadata only; no visual gender classifier is used.
# The Commons resolver still rejects images that do not expose a reusable license.
EXTRA_CURATED_FEMALE_DOMAIN = (
    benchmark.CuratedPortrait("anna_fisher", "Anna Lee Fisher", "Anna Fisher NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("rhea_seddon", "Rhea Seddon", "Rhea Seddon NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("shannon_walker", "Shannon Walker", "Shannon Walker NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("cady_coleman", "Cady Coleman", "Cady Coleman NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("janice_voss", "Janice E. Voss", "Janice Voss NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("pamela_melroy", "Pamela Melroy", "Pamela Melroy NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("laurel_clark", "Laurel Clark", "Laurel Clark NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("kalpana_chawla", "Kalpana Chawla", "Kalpana Chawla NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("judith_resnik", "Judith Resnik", "Judith Resnik NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("tamara_jernigan", "Tamara Jernigan", "Tamara Jernigan NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("linda_godwin", "Linda M. Godwin", "Linda Godwin NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("mary_ellen_weber", "Mary Ellen Weber", "Mary Ellen Weber NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("ellen_baker", "Ellen S. Baker", "Ellen Baker NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("yvonne_cagle", "Yvonne Cagle", "Yvonne Cagle NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("barbara_morgan", "Barbara Morgan", "Barbara Morgan NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("margaret_hamilton", "Margaret Hamilton", "Margaret Hamilton NASA portrait", "adult woman; historical NASA portrait"),
    benchmark.CuratedPortrait("nancy_grace_roman", "Nancy Grace Roman", "Nancy Grace Roman NASA portrait", "adult woman; historical NASA portrait"),
    benchmark.CuratedPortrait("poppy_northcutt", "Frances Northcutt", "Poppy Northcutt NASA portrait", "adult woman; historical space-program portrait"),
    benchmark.CuratedPortrait("joann_morgan", "JoAnn H. Morgan", "JoAnn Morgan NASA portrait", "adult woman; historical NASA portrait"),
    benchmark.CuratedPortrait("kitty_obrien_joyner", "Kitty O'Brien Joyner", "Kitty O'Brien Joyner NASA portrait", "adult woman; historical NACA/NASA portrait"),
    benchmark.CuratedPortrait("pearl_young", "Pearl I. Young", "Pearl Young NACA portrait", "adult woman; historical NACA portrait"),
    benchmark.CuratedPortrait("christine_darden", "Christine Darden", "Christine Darden NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("valerie_thomas", "Valerie Thomas", "Valerie Thomas NASA portrait", "adult woman; NASA institutional portrait"),
    benchmark.CuratedPortrait("mary_w_jackson", "Mary W. Jackson", "Mary Jackson NASA engineer portrait", "adult woman; historical NASA portrait"),
    benchmark.CuratedPortrait("annie_jump_cannon", "Annie Jump Cannon", "Annie Jump Cannon portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("henrietta_leavitt", "Henrietta Swan Leavitt", "Henrietta Swan Leavitt portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("cecilia_payne", "Cecilia Payne-Gaposchkin", "Cecilia Payne Gaposchkin portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("lise_meitner", "Lise Meitner", "Lise Meitner portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("emmy_noether", "Emmy Noether", "Emmy Noether portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("chien_shiung_wu", "Chien-Shiung Wu", "Chien-Shiung Wu portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("maria_goeppert_mayer", "Maria Goeppert Mayer", "Maria Goeppert Mayer portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("gertrude_elion", "Gertrude B. Elion", "Gertrude Elion portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("gerty_cori", "Gerty Cori", "Gerty Cori portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("barbara_mcclintock", "Barbara McClintock", "Barbara McClintock portrait", "adult woman; historical scientific portrait"),
    benchmark.CuratedPortrait("rita_levi_montalcini", "Rita Levi-Montalcini", "Rita Levi Montalcini portrait", "adult woman; scientific portrait"),
    benchmark.CuratedPortrait("florence_nightingale", "Florence Nightingale", "Florence Nightingale photograph", "adult woman; 19th-century photographic portrait"),
    benchmark.CuratedPortrait("clara_barton", "Clara Barton", "Clara Barton photograph", "adult woman; 19th-century photographic portrait"),
    benchmark.CuratedPortrait("susan_b_anthony", "Susan B. Anthony", "Susan B Anthony photograph", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("elizabeth_cady_stanton", "Elizabeth Cady Stanton", "Elizabeth Cady Stanton photograph", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("ida_b_wells", "Ida B. Wells", "Ida B Wells portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("helen_keller", "Helen Keller", "Helen Keller portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("jane_addams", "Jane Addams", "Jane Addams portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("mary_mcleod_bethune", "Mary McLeod Bethune", "Mary McLeod Bethune portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("anna_julia_cooper", "Anna J. Cooper", "Anna Julia Cooper portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("alice_paul", "Alice Paul", "Alice Paul portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("jeannette_rankin", "Jeannette Rankin", "Jeannette Rankin portrait", "adult woman; historical photographic portrait"),
    benchmark.CuratedPortrait("frances_perkins", "Frances Perkins", "Frances Perkins portrait", "adult woman; historical government portrait"),
    benchmark.CuratedPortrait("edith_clarke", "Edith Clarke", "Edith Clarke engineer portrait", "adult woman; historical engineering portrait"),
    benchmark.CuratedPortrait("elsie_macgill", "Elsie MacGill", "Elsie MacGill portrait", "adult woman; historical engineering portrait"),
    benchmark.CuratedPortrait("beatrice_hicks", "Beatrice Hicks", "Beatrice Hicks engineer portrait", "adult woman; historical engineering portrait"),
)


CURATED_80_PLUS = observed.CURATED_FEMALE_DOMAIN + EXTRA_CURATED_FEMALE_DOMAIN


def main() -> int:
    urllib.request.urlopen = _resilient_urlopen
    benchmark.urllib.request.urlopen = _resilient_urlopen
    benchmark.CURATED_FEMALE_DOMAIN = CURATED_80_PLUS
    benchmark.make_scenarios = observed._observed_make_scenarios
    benchmark.run_domain_benchmark = observed._observed_run_domain_benchmark
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
