Rules = """- Compositions with a calculated average atomic size difference greater than 12% across all constituent elements score high, as they are more likely to form amorphous structures.
- Alloys with five or more principal elements, indicating a high mixing entropy, score highest.
- Compositions with a majority of element pairs having a negative heat of mixing score higher, as they are more likely to form a stable amorphous phase.
- Compositions with a high modulus contrast among constituents, particularly with elements known for high Young's modulus, e.g., Ti and some rare earths like Gd, score higher.
- Alloys with a Tg/Tl ratio closer to or greater than 0.6 score highest, as they demonstrate better thermal stability.
-  Compositions with a balanced ratio of late transition metals, particularly when combined with Zr or Ti, score higher, as these are conducive to forming amorphous structures.
- Alloys with a greater variety of elements, especially those that introduce complexity without significantly increasing the tendency for crystalline phase formations, score higher.
"""

EvalueSystem = "You will act as an expert in materials science, specializing in bulk metallic glass (BMG). Your task is to analyze provided data points, which are based on specific screening rules for BMGs components."

EvalueUser = """Firstly, understand and apply the following screening rules defined as RULE. These rules are crucial for evaluating the potential of each BMG component.I will also provide you with performance data for Real BMGs similar to the data in the DATA, which are actually measured and can be used as a reference for your evaluation.

Secondly, evaluate each data point in DATA by assigning a score from 0 to 1, indicating its suitability for experimental validation in bulk metallic glass (BMG). A score of 1 signifies high relevance. Include a short explanation for each score, focusing on pertinent scientific concepts and considering the complexities and potential results of the experimental process in HEAs.

RULE:
{rule}

Real BMGs:
{sim}

DATA:
{data}

Ensure the output for each evaluated data point is formatted as follows, data calculations can be analyzed if necessary. The format of the rating needs to be output in the following format, additional calculations are allowed, but the rating information needs to be wrapped in `{{}}`:
{{
    "score": [assigned value],
    "reason": [brief overview explaining the assigned value]
}}"""