# Annotation guidelines -- tool-call hallucination study

You will see a user request, the schema of the available function(s) and a tool call produced by a
language model. Judge the call ONLY against the request and the schema. Work alone; do not discuss
items with the other rater until both of you have finished.

## Pass 1 (sheet pass1_raterN.csv) -- do this first
* `call_is_wrong`: `yes` if the call would not correctly do what the user asked, else `no`.
* `error_types`: if wrong, one or more of the codes below, separated by `;`
  * WRONG_FUNCTION -- names a function that does not exist, or the wrong one of the available functions
  * EXTRA_PARAMETER -- passes a parameter the function does not declare
  * MISSING_PARAMETER -- leaves out a parameter that the request needs or the schema requires
  * WRONG_TYPE -- right parameter, value of the wrong type (e.g. "10" where a number is declared)
  * WRONG_VALUE -- right parameter and type, but the value does not match the request
  * WRONG_FORMAT -- not a single {"function": ..., "parameters": {...}} JSON object
  * OTHER -- wrong for a reason none of the above covers (explain in notes)
* `correct_call`: if wrong, write the call that SHOULD have been made, as JSON
  {"function": "...", "parameters": {...}}. Leave empty if the call is right.

## Pass 2 (sheet pass2_raterN.csv) -- only after finishing pass 1
* `corrected_call_valid`: `yes` if the corrected call is a well-formed call that matches the schema.
* `corrected_call_intent`: `yes` / `partly` / `no` -- does the corrected call do what the user asked?

## Edge cases (agree on more during the practice round and add them here)
* An empty string "" for an optional parameter means "not set"; it is not an error.
* Numbers written as strings ("5") count as WRONG_TYPE only if the schema declares a number.
* A value that is reasonable but not exactly what the user typed (e.g. "NYC" for "New York") is NOT wrong
  unless the API would clearly misread it.
* If the request needs two calls, judge whether this call is a correct FIRST step.
