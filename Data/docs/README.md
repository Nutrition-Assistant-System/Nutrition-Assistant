# Data quality log

Generated 2026-11-12 from the data files. Rules: no value is estimated, imputed or corrected. Checks either
**flag** a record (USDA: values kept) or **drop** it (Open Food Facts: the record is removed, never edited).
A blank cell means "not reported by the source", never zero.

## USDA FoodData Central

### Missing core nutrients (14 of 208 foods)

| fdc_id | Food | Not reported |
|---|---|---|
| 175120 | Atlantic mackerel (cooked) | total sugars |
| 173694 | Sea bass (cooked) | total sugars |
| 175180 | Shrimp (cooked) | total sugars, fiber |
| 175179 | Shrimp (raw) | total sugars, fiber |
| 168191 | Dates, Medjool (dried) | saturated fat, cholesterol |
| 169699 | Couscous (raw) | total sugars |
| 168874 | Quinoa (raw) | total sugars |
| 169705 | Rolled oats (raw) | total sugars |
| 173746 | Navy (white) beans (cooked) | cholesterol |
| 173745 | Navy (white) beans (raw) | cholesterol |
| 174284 | Red lentils (raw) | total sugars |
| 174313 | Lamb leg (raw) | total sugars |
| 170554 | Chia seeds (dried) | total sugars |
| 173475 | Spearmint (fresh mint) (raw) | total sugars |

| Nutrient | Foods with no value |
|---|---|
| saturated fat | 1 |
| total sugars | 11 |
| fiber | 2 |
| cholesterol | 3 |

The assistant must answer "not reported" for these, not 0.

### Flags (`quality_flags`, 21 foods)

| Check | Rule | Result |
|---|---|---|
| Energy | Estimate = 4 x protein + 4 x carbohydrate + 9 x fat. Flag if the published energy differs from the estimate by more than 15% **and** more than 5 kcal (same rule for every food) | 14 foods, mostly vegetables: USDA uses food-specific energy factors, so the gap is expected. Values kept |
| Sugars vs carbohydrate | Flag if total sugars > carbohydrate by more than 0.01 g | 6 dairy foods: sugars are measured directly, carbohydrate "by difference". Values kept |
| Main components | Flag if water + protein + fat + carbohydrate + ash is outside 97 to 103 g | 1 food; separately, 14 foods sum to between 101 and 103 g, which is within the rule and normal for separate lab methods |

| fdc_id | Food | quality_flags |
|---|---|---|
| 171304 | Greek yogurt, plain, whole milk (processed) | sugars_gt_carbohydrate |
| 170859 | Heavy cream (processed) | sugars_gt_carbohydrate |
| 170872 | Milk, low fat (1%) (processed) | sugars_gt_carbohydrate |
| 171267 | Milk, reduced fat (2%) (processed) | sugars_gt_carbohydrate |
| 171269 | Milk, skim (nonfat) (processed) | sugars_gt_carbohydrate |
| 171265 | Milk, whole (3.25% fat) (processed) | sugars_gt_carbohydrate |
| 175120 | Atlantic mackerel (cooked) | proximates_sum_96.5g |
| 167747 | Lemon juice (raw) | energy_vs_4-4-9_gap_+42% |
| 169967 | Broccoli (cooked) | energy_vs_4-4-9_gap_+20% |
| 170379 | Broccoli (raw) | energy_vs_4-4-9_gap_+21% |
| 169986 | Cauliflower (raw) | energy_vs_4-4-9_gap_+20% |
| 169229 | Eggplant (cooked) | energy_vs_4-4-9_gap_+15% |
| 169261 | Okra (cooked) | energy_vs_4-4-9_gap_+25% |
| 169260 | Okra (raw) | energy_vs_4-4-9_gap_+19% |
| 170416 | Parsley (raw) | energy_vs_4-4-9_gap_+23% |
| 173475 | Spearmint (fresh mint) (raw) | energy_vs_4-4-9_gap_+21% |
| 168463 | Spinach (cooked) | energy_vs_4-4-9_gap_+27% |
| 168462 | Spinach (raw) | energy_vs_4-4-9_gap_+28% |
| 169999 | Sweet corn, yellow (cooked) | energy_vs_4-4-9_gap_+16% |
| 169998 | Sweet corn, yellow (raw) | energy_vs_4-4-9_gap_+16% |
| 169251 | White mushrooms (raw) | energy_vs_4-4-9_gap_+29% |

### Other notes
- `fdc_publication_date` (2019-04-01) is when USDA loaded SR Legacy into FoodData Central; the data is the April 2018 release.
- Columns that SR Legacy leaves empty were removed: `measure_unit` and `portion_description` (portions; the unit is written in `modifier`, for example "cup, chopped") and `median` (nutrient table). `footnote` is kept (filled in 17 nutrient rows and 0 portion rows).
- 1 food has no household portion in SR Legacy: Greek yogurt, plain, whole milk.

## Open Food Facts

### Products removed, by reason (script v2)

| Reason | Products removed |
|---|---|
| not_in_selected_categories | 4630113 |
| no_ingredient_list | 67053 |
| no_brand | 39399 |
| incomplete_nutrition | 15389 |
| duplicate_brand_and_name | 9960 |
| no_product_name | 9514 |
| excluded_by_category_rules | 6537 |
| energy_inconsistent_with_macros | 2927 |
| duplicate_ingredients_and_nutrients | 2484 |
| salt_sodium_mismatch | 1639 |
| salt_above_category_cap | 1021 |
| no_country | 896 |
| macros_exceed_100g | 322 |
| value_out_of_range | 283 |
| saturated_fat_gt_fat | 266 |
| sugars_gt_carbohydrates | 176 |
| name_not_english | 87 |
| ingredients_not_english | 59 |
| duplicate_barcode | 1 |

### Rules applied (products are dropped, never edited)
- **Category:** first project category whose include tags match, unless an exclude tag or exclude word in the name matches (for example dairy desserts are not yogurts, rice cakes are not bread, pastries are not biscuits, squash concentrates are not drinks).
- **Required:** product name, brand, country, ingredient list, and energy, fat, saturated fat, carbohydrate, sugars, protein and salt.
- **Ranges:** every value 0 to 100 g, energy 0 to 900 kcal.
- **Salt cap per category (g per 100 g/ml):** breakfast cereals 3, yogurts 1, chocolate 1.5, biscuits and salty snacks 5, soft drinks and juices 0.5, bread 3. These are data-plausibility limits, not nutrition advice.
- **Salt vs sodium:** when both are published, salt must equal sodium x 2.5 within 10% (at least 0.1 g).
- **Saturated fat <= fat** and **sugars <= carbohydrate** (0.001 g tolerance). Fat + carbohydrate + protein <= 101 g.
- **Energy:** within 20% (at least 15 kcal) of 4 x protein + 4 x carbohydrate + 9 x fat + 2 x fibre.
- **Duplicates:** same barcode; same brand + name; same ingredient text + same nutrient values (the best-ranked copy is kept).
- **Language:** ingredient list English by a word-frequency vote (at least 75% English words, or 50% plus langdetect agreeing); product name rejected if it contains a clearly foreign word, more foreign than English words, or non-Latin script. Brand-only names count as neutral.
- **Ranking:** test-set products first, then sold in an English-speaking country, allergen tags present, serving size, fibre, completeness, most recent edit.

### Remaining known issues
- **Fibre above carbohydrate** in 1 product(s) (flag `fiber_gt_carbohydrates`). On EU labels carbohydrate excludes fibre, so this can be genuine (for example very dark chocolate).
- **Rounded or computed values:** for labels that declare values per serving, Open Food Facts computes per-100 g values, which gives long decimals. Values are kept as published; text shown to the model is rounded.
- **0 kcal drinks:** labelling rules allow "0 kcal" below about 4 kcal per serving. Kept.
- **Free-text fields** (`quantity`, `serving_size`) are kept as printed and used only for display; calculations use the numeric `serving_quantity`.
- **Allergen tags** are crowd-sourced: an empty field does not prove a product is allergen-free.
