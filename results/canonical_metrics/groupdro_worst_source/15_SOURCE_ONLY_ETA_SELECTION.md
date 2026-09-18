# Source-only eta selection

```json
{
  "all_eta_summaries": [
    {
      "eta": 0.01,
      "held_position_means": {
        "P1": 0.18986130555221375,
        "P2": 0.18344203887678096,
        "P3": 0.1311531212622858
      },
      "mean_held_source_accuracy": 0.20584126984126982,
      "mean_held_source_macro_f1": 0.16815215523042687,
      "mean_max_q_weight": 0.5144108474281518,
      "minimum_held_position_mean_macro_f1": 0.1311531212622858,
      "population_sd_held_position_seed_macro_f1": 0.04432745246580674,
      "q_weight_stability_population_sd": 0.011910698916064241,
      "source_worst_position_macro_f1": 0.1311531212622858
    },
    {
      "eta": 0.05,
      "held_position_means": {
        "P1": 0.17035194714604138,
        "P2": 0.1791189200027374,
        "P3": 0.13545419425832328
      },
      "mean_held_source_accuracy": 0.20687830687830688,
      "mean_held_source_macro_f1": 0.16164168713570068,
      "mean_max_q_weight": 0.5257050716432022,
      "minimum_held_position_mean_macro_f1": 0.13545419425832328,
      "population_sd_held_position_seed_macro_f1": 0.038600484161418966,
      "q_weight_stability_population_sd": 0.018149627121991405,
      "source_worst_position_macro_f1": 0.13545419425832328
    },
    {
      "eta": 0.1,
      "held_position_means": {
        "P1": 0.14479026524788652,
        "P2": 0.17440977817635026,
        "P3": 0.11152644111567836
      },
      "mean_held_source_accuracy": 0.18855026455026455,
      "mean_held_source_macro_f1": 0.1435754948466384,
      "mean_max_q_weight": 0.5356238587262883,
      "minimum_held_position_mean_macro_f1": 0.11152644111567836,
      "population_sd_held_position_seed_macro_f1": 0.038331820023972396,
      "q_weight_stability_population_sd": 0.02180594088098645,
      "source_worst_position_macro_f1": 0.11152644111567836
    },
    {
      "eta": 0.2,
      "held_position_means": {
        "P1": 0.18382821881786932,
        "P2": 0.17752589389883502,
        "P3": 0.12849308887404426
      },
      "mean_held_source_accuracy": 0.20196825396825396,
      "mean_held_source_macro_f1": 0.16328240053024956,
      "mean_max_q_weight": 0.537052021715971,
      "minimum_held_position_mean_macro_f1": 0.12849308887404426,
      "population_sd_held_position_seed_macro_f1": 0.04590123743674114,
      "q_weight_stability_population_sd": 0.03802436742316132,
      "source_worst_position_macro_f1": 0.12849308887404426
    }
  ],
  "best_mean_held_source_macro_f1": 0.16815215523042687,
  "eligibility_threshold": 0.15815215523042686,
  "eligible_etas": [
    0.01,
    0.05,
    0.2
  ],
  "frozen_selection_receipt_sha256": "9bb41bd2996090c04d61ae15bdd98754d9804c22496ad297a87e9af56d744f97",
  "minimum_position_tied_etas": [
    0.01,
    0.05
  ],
  "sd_tied_etas": [
    0.05
  ],
  "selected_eta": 0.05,
  "selection_frozen_at_utc": "2026-08-05T03:19:05.472261+00:00",
  "selection_input_p4_accessed": false,
  "selection_sha256": "02c5eed9ba13cae494c2a4c81ef660ca42f2fce063008a9592db0cd424a345cd",
  "selector": "frozen_lexicographic_source_only_v1"
}
```
