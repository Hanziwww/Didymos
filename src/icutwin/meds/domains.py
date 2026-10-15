"""MEDS code domains in canonical order (S4): an event's domain rank orders events of one time (and the
vocabulary's domain tokens). A module of its own so the tokenizer and the samplers need not import the build."""

DOMAIN_ORDER = ("PATIENT", "ICU", "HOSP", "ADT", "VITAL", "HEMO", "RESP", "MCS", "CRRT", "IHD", "LAB", "MICRO_ORDER",
                "MICRO", "MICRO_NC", "MICRO_SUS", "MED_CON", "MED_INT", "MED_ORD", "INPUT", "OUTPUT", "PA", "POS",
                "AIRWAY", "LINE", "DRAIN", "CODE_STATUS", "PROC", "PT_DX", "HOSP_DX")
