# SFT model on 8 validation questions (greedy, same settings as the test)

**Q:** Which has more fat per 100 g: tahini or chia seeds?

- **Reference:** Tahini: 53.76 g per 100 g, compared with 30.74 g for chia seeds (USDA FoodData Central).
- **Model:** Chia seeds: 3.5 g per 100 g, compared with 1.2 g for tahini (USDA FoodData Central).
- stopped at <|end|>: True (30 tokens)

**Q:** Which has more fat per 100 g: hummus, commercial or cooked fava beans (broad beans)?

- **Reference:** Hummus, commercial: 17.82 g per 100 g, compared with 0.4 g for cooked fava beans (broad beans) (USDA FoodData Central).
- **Model:** Hummus, commercial: 3.5 g per 100 g, compared with 1.2 g for cooked fava beans (broad beans) (USDA FoodData Central).
- stopped at <|end|>: True (36 tokens)

**Q:** Which has more fat per 100 g: raw tilapia or tuna, light, canned in water?

- **Reference:** Raw tilapia: 1.7 g per 100 g, compared with 0.82 g for tuna, light, canned in water (USDA FoodData Central).
- **Model:** Raw tilapia: 3.5 g per 100 g, compared with 1.2 g for tuna, light, canned in water (USDA FoodData Central).
- stopped at <|end|>: True (35 tokens)

**Q:** Which has fewer calories per 100 g: cooked yellowfin tuna or anchovy, canned in oil?

- **Reference:** Cooked yellowfin tuna: 130 kcal per 100 g, compared with 210 kcal for anchovy, canned in oil (USDA FoodData Central).
- **Model:** Cooked yellowfin tuna: 0 kcal per 100 g, compared with 0 kcal for anchovy, canned in oil (USDA FoodData Central).
- stopped at <|end|>: True (31 tokens)

**Q:** Which has more carbohydrate per 100 g: apricots, dried or raw apricot?

- **Reference:** Apricots, dried: 62.64 g per 100 g, compared with 11.12 g for raw apricot (USDA FoodData Central).
- **Model:** Raw apricot: 3.5 g per 100 g, compared with 1.2 g for apricots, dried (USDA FoodData Central).
- stopped at <|end|>: True (33 tokens)

**Q:** How much sugar is in 100 g of sour cream?

- **Reference:** 100 g of sour cream contains 3.41 g of sugar (USDA FoodData Central).
- **Model:** I don't have that information: sour cream is not in my nutrition data, so I can't give a reliable value.
- stopped at <|end|>: True (26 tokens)

**Q:** What is the potassium content of raw chicken, whole (meat only) per 100 g?

- **Reference:** 100 g of raw chicken, whole (meat only) contains 229 mg of potassium (USDA FoodData Central).
- **Model:** 100 g of raw chicken, whole (meat only) contains 0 mg of potassium (USDA FoodData Central).
- stopped at <|end|>: True (24 tokens)

**Q:** What is the sodium content of white bread per 100 g?

- **Reference:** 100 g of white bread contains 490 mg of sodium (USDA FoodData Central).
- **Model:** 100 g of white bread contains 0 mg of sodium (USDA FoodData Central).
- stopped at <|end|>: True (18 tokens)

