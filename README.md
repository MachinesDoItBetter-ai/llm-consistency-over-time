# llm-consistency-over-time
Have you ever wondered how consistent LLM results are over time? Noticed how performance seems to degrade in peak times? In this article we explore the consistency of LLMs over time by asking several free LLMs (ChatGPT, Grok, and Gemini) the same question once per day for a week.

## Experiment Parameters

### The Question

If we ask a quesiton that is easy it is likely we get the same answer each time, so we wanted to choose a question that was quite hard. We also wanted a question which had a very clear correct answer and a question for which the answer could be partially right, for example a list where you could have some items from the correct anwser but not others. We eventually arrived at the following prompt:

```
Can you list the best 5 scrabble words I can make from these letters assuming it is the first move. The word will get double score, and if long enough it will get the double letter bonus on one of the letters. Consider this position and report only the best score.
  s
Do not provide any explanation, nor position, only the list of words and scores, with one word per line followed by its score.

Letters: BADIERK

```

THis quesiton invovles some complex reasoning with which the LLMs will likely struggle, positiioining the word affects which letter gets the double score. But it also has plenty of potential for paartially correct answerss (e.g. correct words, wrong score). The prompt includes a specification of an anwer format, another place where we can test the consitency of the LLM repsonses.

The correct answer to our prompt is:

```

```
### Models tested

## Experiment Results


### Accuracy

Although the goal of the experiemnt was not to test the accuracy of the responses, its important that we consider the accuracy of the answers, else we could reward a repsonse that was nonsensical but consistent (e.g. "No I can't do that").



seful to have some understan
Note that we are not trying to 

Comparing LLM responses to same query over time.

Intro

Methdology

Prompt

Scoring

Details

Repo Link