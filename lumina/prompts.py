# -*- coding: utf-8 -*-
"""All prompts for LUMINA aqua + wildfire extraction and cross-validation.

Prompt structure (Stage 1: Examiner):
  Each LLM call sends 3 messages in sequence:
    1. System prompt    → message_system_v2 (role + domain context)
    2. User instruction → message_system_v2_output (output format + paper content)
    3. User question    → domain-specific question (JSON schema + constraints)

Prompt structure (Stage 4: Cross-Validation):
  Each LLM call sends 2 messages:
    1. System prompt    → message_system_ragQuery (evidence verifier role)
    2. User instruction → checker_requery / checker_requeryFull (context + answer + rules)
"""

# ---- Stage 1: Examiner — System Prompt (消息 1) ----
# Sets the LLM's role as a domain expert scientist reading the paper
# {domain} is replaced with the domain_knowledge string from config
message_system_v2 = """
You are skilled in Chinese/English paper reading; You are also a scientist and expert in {domain}. 
You read through the whole paper (from the beginning to the end);
provide the best answers (may include multiple items) you can find to the question that I ask.
"""

# ---- Stage 1: Examiner — Output Instruction + Paper Content (消息 2) ----
# Instructs the LLM to output JSON with value, evidence, and confidence_lv
# {content} is replaced with the full markdown text (truncated before References)
# Forces JSON output via response_format={"type": "json_object"} in the API call

message_system_v2_output = """
Read the question, analyze step by step, provide your answer and your confidence (0% to 100%) to this answer. 
Also provide the direct quote (evidence) when required.
Note: 
1. The direct quote indicates the specific textual evidence from the article that supports your conclusions.
2. Provide direct quotes of all evidences. If the evidence is in a table or figure, directly reference the table name or figure name. 
3. The confidence indicates how likely you think your answer is true. Note that the confidence level should be high if you insist that there are no required values.

Please read this markdown content:\n{content}
"""

# ---- Stage 4: Cross-Validation — System Prompt (消息 1) ----
# Positions the LLM as an evidence verifier, not a strict fact-checker
# The task is to confirm whether a piece of evidence exists in the retrieved context
message_system_ragQuery = """
You are a scientific evidence verifier rather than a strict semantic fact-checker. 
You are going to check:
If the provided sentence, paragraph, or table closely aligns with the most relevant texts measured by vectorizations, return Yes if so, return No if not.
"""

# ---- Stage 4: Cross-Validation — Verification Task (消息 2) ----
# {context}  = the retrieved chunk(s) from the original paper
# {key_topic} = the item name (e.g. "Study_location", "Forest_smoldering")
# {answer}    = the evidence text from one model's examiner output
# The LLM must return JSON: {"existing_flag": 0|1, "direct_quote": "..."}

checker_requery = """
Rethink before you do the checker and then proceed with the following:

Your task is to verify whether the "answer" in the Candidate Response
is supported as an existing piece of evidence in the Original Context,
even if the support is indirect, structural, or coarse-grained.

### Original Context (Source):
{context}

### Candidate Response (Target):
Our answer to the {key_topic} is {answer}.

### Instructions:
1. Ignore conversational fillers (e.g., "Our answer to... is...").
2. Focus on whether the stated "answer" (e.g., a table, figure, section, or dataset)
   **exists in the Original Context and is topically related to the {key_topic}**.
3. Do NOT require the answer to contain explicit quantitative results
   for the {key_topic} unless explicitly stated.
4. If the answer exists in the Original Context (e.g., as a table caption, section title,
   or referenced evidence), set:
   - "existing_flag" = 1
   - "direct_quote" = the exact text 
5. Only set "existing_flag" = 0 if the answer does NOT appear anywhere
   in the Original Context as a named or referenced evidence item.

### Constraints:
- The "direct_quote" must be copied 100% verbatim from the Original Context.
- Table captions, figure captions, section headers, and in-text references
  are all valid sources of evidence.
  
Example:
{{
    'existing_flag': 0,
    'direct_quote': N/A,
}}
"""

checker_requeryFull = """
Rethink before you do the checker and then proceed with the following:

Your task is to verify whether the "answer" in the Candidate Response
is supported as an existing piece of evidence in the Original Context,
even if the support is indirect, structural, or coarse-grained.

### Original Context (Source):
{context}

### Candidate Response (Target):
Our answer is {answer}.

### Instructions:
1. Ignore conversational fillers (e.g., "Our answer to... is...").
2. Focus on whether the stated "answer" (e.g., a table, figure, section, or dataset)
   **exists in the Original Context**.
3. Do NOT require the answer to contain explicit quantitative results unless explicitly stated.
4. If the answer exists in the Original Context (e.g., as a table caption, section title,
   or referenced evidence), set:
   - "existing_flag" = 1
   - "direct_quote" = the exact text 
5. Only set "existing_flag" = 0 if the answer does NOT appear anywhere
   in the Original Context as a named or referenced evidence item.

### Constraints:
- The "direct_quote" must be copied 100% verbatim from the Original Context.
- Table captions, figure captions, section headers, and in-text references
  are all valid sources of evidence.
  
Example:
{{
    'existing_flag': 0,
    'direct_quote': N/A,
}}
"""

# -------------------------
# Aqua domain prompts (3 questions)
# -------------------------

# Q1: Study metadata — location, period, coordinates
# Extracted as meta items (text), ensembled via ensemble_utils_meta
aqua_question_A_meta = """
What are the study locations and study period in this study? Answer the above question and provide direct evidence.
Note: 
1. 'Study locations' refer to the regions that were the focus or sites of this research. 
2. 'Study period' refer to the specific duration or timeframe during which the research or investigation is conducted. 
It indicates the time allocated for data collection, analysis, or other activities related to the study.
3. Evidence refers to directly quoting the exact text or the titles of tables from the article. Do not perform any paraphrasing. 
Directly extract the content (full sentence or the Table titles) from the article that supports your answer.
4. The latitude and longitude should be in decimal format
The answer should be provided exclusively in JSON format, following the example structure below:
{
    'Study_location': 
    {
        'value': Location;
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
    'Study_location_detail': 
    {
        'value': Province, City;
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
    'Study_period': 
    {
        'value': 2003-2017;
        'evidence': textual evidence or table title;
        'confidence_lv': 80;
    }
    'Latitude':
    {
        'value': 45°N;
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
    'Longitude':
    {
        'value': 130°E;
        'evidence': textual evidence or table title;
        'confidence_lv': 90;
    }
    
}
"""

# Q2: Cultured species — constrained choice (fish/shrimp/crab/mixed/others)
# Extracted as meta item (text), ensembled via ensemble_utils_meta
aqua_question_B_experiment = """
Which of the following species were cultured in the aquaculture ponds where this study measured CH4 flux? The value should be chosen only from the following selections:
A [fish] B [shrimp] C [crab] D [mixed] E [others].

Answer the above question and provide direct evidence.
Note:
1. Evidence refers to directly quoting the exact text or the titles of tables from the article. Do not perform any paraphrasing. 
Directly extract the content (full sentence or the Table titles) from the article that supports your answer.
2. "mixed" refers to the culturing of multiple species in a single pond, for example, fish, shrimp, and crab.
3. The above question refers to the species cultured in the aquaculture ponds of this study, excluding those from other research.

The answer should be provided exclusively in JSON format, following the example structure below:
{
    'Specie': 
    {
        'value': A [fish];
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
}
"""

# Q3: Methane flux values — per-test extraction, may have multiple items
# Extracted as numeric items, ensembled via ensemble_utils_value
aqua_question_C_flux = """
What are the methane flux values for each of the comparative tests (e.g. different aquaculture ponds, experimental treatments, etc) measured in this study ? 
Answer the above question and provide direct evidence.

Note:
1. Extract only the methane flux from aquaculture ponds measured in this study. Flux refers to greenhouse gas emissions per unit time per unit area. 
Do not look only in the abstract; the answer is usually in the results section.
4. Evidence refers to directly quoting the exact text or the titles of tables from the article. Do not perform any paraphrasing. 
Directly extract the content (full sentence or the Table titles) from the article that supports your answer.


The answer should be provided exclusively in JSON format and may include multiple items, following the example structure below:
{
    'Flux-1': 
    {
        'value': 2. 12;
        'evidence': textual evidence or table title;
        'confidence_lv': 95;        
        'unit': mg・m¯².h¯¹;
    }
    ...
}

"""

# -------------------------
# Wildfire domain prompts (4 questions)
# -------------------------

# Q1: Study metadata — location, period
# Extracted as meta items, ensembled via ensemble_utils_meta
wildfire_question_A_meta = """
What are the study locations and study period in this study? 
Answer the above question and provide direct evidence.
Note: 
1. 'Study locations' refer to the regions that were the focus or sites of this research.
2. 'Study period' refer to the specific duration or timeframe during which the research or investigation is conducted. 
It indicates the time allocated for data collection, analysis, or other activities related to the study.
3. Evidence refers to directly quoting the exact text or the titles of tables from the article. 
Do not perform any paraphrasing. Directly extract the content (full sentence or the Table titles) from the article that supports your answer.
When there are no relevant information, the confidence_lv shoule be set to -1

The answer should be provided exclusively in JSON format, following the example structure below:
{
    'Study_location': 
    {
        'value': name(s) of the study area;
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
    'Study_period': 
    {
        'value': 2003-2017;
        'evidence': textual evidence or table title;
        'confidence_lv': 100;
    }
}
"""

# Q2-4: Emission factors — parameterized by gas type (co2/ch4/n2o)
# Each fuel × combustion type is a separate item (e.g. Forest_smoldering)
# Extracted as numeric items, ensembled via ensemble_utils_value
# {emission} is replaced by the gas name via wildfire_question_EFQuery()
wildfire_question_B_ef_details = """
What are the emission factors for {emission} associated with various types of fuels? Answer the above question and provide direct evidence. 

Note: 
1. Emission factors: The amount of a specific greenhouse gas released when a unit of fuel is burned or consumed. 
Usually measured in mass (e.g., g of {emission} per kg of dry mass). Depends on the type of fuel and how it is used.
2. Various types of fuels: Different materials used as energy sources, such as forests, peatland, crop residues. 
List None of these examples if there are no relevant information. 
If the fuels are mentioned yet without relevant information, the value should be set to 999999 and confidence_lv set to -1.
3. Evidence refers to directly quoting the exact text or the titles of tables from the article. Do not perform any paraphrasing. 
Directly extract the content (full sentence or the Table titles) from the article that supports your answers. 
4. Find the combustion condition, use results only from the following selections (smoldering, flamming, mixed, None) 
Put it in the name as a unique (e.g, Forest_smoldering or Forest_flaming or Forest_mixed or Forest_None)
5. Fill in the MCE value (measured combustion efficiency) if it is mentioned in the paper, otherwise None
6. Note that for experimental research paper, there may be multiple times of experimental results on the same fuel with different estimates, 
list all of them with markers in experimental key. 
7. Mark the key of experimental as True if the value is from experiments, mark the key of experimental as Ref if the value is from references.

The answer should be provided exclusively in JSON format, following the example structure below:
{{
    'Forest_smoldering':
    {{
        'value': 1675;
        'evidence': Table 2, Emission factors for forest combustion;
        'confidence_lv': 95;
        'mce': None,
        'experimental': Ref
    }}
    'Forest_flamming':
    {{
        'value': 1755;
        'evidence': Table 2, Emission factors for forest combustion;
        'confidence_lv': 95;
        'mce': None,
        'experimental': Ref
    }}
    ...
}}
"""


def wildfire_question_EFQuery(emission: str) -> str:
    return wildfire_question_B_ef_details.format(emission=emission)


def questions_for_domain(domain: str):
    d = domain.lower()
    if d.startswith("aqua"):
        return [aqua_question_A_meta, aqua_question_B_experiment, aqua_question_C_flux]
    if d.startswith("wild"):
        return [
            wildfire_question_A_meta,
            wildfire_question_EFQuery("co2"),
            wildfire_question_EFQuery("ch4"),
            wildfire_question_EFQuery("n2o"),
        ]
    raise ValueError(f"Unsupported domain: {domain}. Expected aqua or wildfire.")


def emission_type_for_question(domain: str, question_index: int):
    if domain.lower().startswith("wild") and question_index > 1:
        return {2: "co2", 3: "ch4", 4: "n2o"}.get(question_index)
    return None
