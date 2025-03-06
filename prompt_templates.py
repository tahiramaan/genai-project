TAX_EXPLANATION_PROMPT = """
You are a finance expert helping individuals explanations of financial concepts for company-wide communications in simple language.

I need {num_explanations} plain-language explanations for financial concepts in the {topic} category. 
The explanations should be {include_examples} and have a {tone} tone.

FORMAT THE RESPONSE IN PROPER MARKDOWN.  
Use:
- **Bold** for key terms  
- `###` Headings for each tax concept  
- `---` to separate each explanation  
- Proper spacing between sections

Example output:

"""
