# Training-data fixes after the AI-assisted hand-check

Date: 12 Nov 2026. Scope: `src/prepare_data.py` (both CPT documents and SFT pairs) and a new validator, `src/validate_data.py`.
No model has been trained yet.

## What did not change
- `data/raw/foods_and_products.csv` (merged source data): unchanged (same SHA-256).
- `tests/test_questions.csv` (frozen test set): unchanged (same SHA-256).
- Train / held-out membership of all 988 CPT documents: unchanged (889 / 99; 0 test items held out).
- Test separation: the same 31 rows (18 USDA foods, 6 products) are excluded from SFT.
- SFT size and mix: 894 pairs, same count per type.

## Fixes (shared by CPT and SFT unless marked)
| # | Fix | CPT | SFT |
|---|---|---|---|
| 1 | Allergen answers: "Yes" if the allergen tags **or** the ingredient list shows the allergen; "No" only when neither does. `_wheat_` markup removed before checking. "May contain" statements are treated as warnings, never as ingredients. The matching ingredient is quoted exactly. | Ingredient text without `_` markup; only standard allergen tags listed | Yes |
| 2 | Rounding: halves round up (4.635 -> 4.64), using exact decimal arithmetic | Yes (displayed values) | Yes |
| 3 | Working uses the displayed numbers, so "10.7 x 3.55 = 37.99" always adds up | n/a | Yes |
| 4 | Names: one `label()` for CPT and SFT; trailing punctuation removed; proper nouns kept ("Atlantic"); no "cooked" prefix when the name says "dry-roasted"; brand not repeated ("Lizi's Lizi's"); company-registration brands (Inc, Llc, Ltd) not used as prefixes; 17 generic names renamed from the source brand field ("Cereal" -> "Post Honey Bunches Of Oats Cereal") | Yes | Yes |
| 5 | Household measures phrased naturally ("1 pita small (4" dia)", "1 cup with pits ... gives 138 g of edible food") | Yes | Yes |
| 6 | Serving size from the numeric `serving_quantity` (printed text has typos such as "1auffin") | Yes | already |
| 7 | Brand line cleaned (no double spaces or "Inc..") | Yes | n/a |
| 8 | Grammar: plural unknown foods ("ramen noodles are"); weight-specific refusal | n/a | Yes |

## Validation (all pairs, all documents)
- `src/validate_data.py` re-derives every SFT answer and every CPT nutrient value from the source table and checks arithmetic, basis, allergen yes/no (independently from tags + ingredients), quoted ingredients, traces and label warnings, test leakage and similarity to test questions.
- Result: **894 pairs and 988 documents checked: 0 critical issues, 0 warnings.**
- Validator self-test: 10 deliberately injected errors (wrong value, wrong result, wrong portion weight, headline mismatch, flipped allergen, "may contain" used as ingredient, wrong serving, test item leaked, "missing" value that exists, refusal giving advice) -> 10 of 10 detected.
- Hand-check re-run on the same 50 random pairs + 9 refusal / don't-know / unknown-food pairs (`results/sft_ai_handcheck_results_v2.csv`): 59 PASS, 0 NEEDS_REVIEW, 0 FAIL (v1: 50 / 8 / 1, kept as `results/sft_ai_handcheck_results_v1_before_fixes.csv`).

## Remaining minor notes (not errors, kept on purpose)
- Some product names are short, truncated or abbreviated in the source ("Tube it", "4 hot dog rolls imp", "Gmills hny nut cheerios ... whl grn"). They were not rewritten, to avoid inventing names.
- One generic name could not be made specific from the source brand field ("Kellogg's Cereal").
- USDA word order is kept in some names ("raw grapes, red or green").
