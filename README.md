# JigsawUnintendedBias
The Conversation AI team, a research initiative founded by Jigsaw and Google (both part of Alphabet), builds technology to protect voices in conversation. Challenge here is to build machine learning models that detect toxicity and reduce unwanted bias. For example, if a certain minority name is frequently associated with toxic comments, some models might associate the presence of the minority name in a message that is not toxic and wrongly classify the comment as toxic.

# Business Constraints & Understanding as a machine learning problem
> Latency requirement is not mentioned. so we consider evaluation time to be a few seconds.
> Some form of interpretibility of the results
> Model Evaluation metric - overall AUC + generalized mean of bias AUC's.

### Model Predicted with a combined AUC score of 0.924.
