# Multi-Agent Debate Evaluation Metrics Implementation Plan

Based on the evaluation metrics documents, this plan outlines features to implement for improving debate quality and evaluation.

## High Priority

### 1. Information Gain Metric
- **Description**: Use cosine dissimilarity between consecutive agent responses to verify they're actually debating vs. falling into repetitive agreement loops
- **Implementation**:
  - Add embedding generation for each agent response
  - Calculate cosine dissimilarity between consecutive responses
  - Track and display information gain scores per round
  - Alert if dissimilarity drops below threshold (indicating repetitive loop)

### 2. Position Swapping for Judge Evaluation
- **Description**: Run judge evaluation twice with swapped Proposer/Critic argument order to prevent position bias, then average the scores
- **Implementation**:
  - Modify judge evaluation to run twice per round
  - Swap the order of Proposer and Critic arguments in the second run
  - Average the consensus scores and verdicts from both runs
  - Store both individual scores and the averaged result

### 3. Judge Profiles
- **Description**: Add selectable system prompts for different judge reasoning styles
- **Profiles to implement**:
  - **Logical Thinker**: Breaks down complex problems into logical steps, focuses on clear reasoning and consistency
  - **Robust Reasoner**: Tackles complex tasks with thorough and resilient reasoning, provides precise justifications
  - **Deductive Reasoner**: Uses deductive logic to derive conclusions from given premises
  - **Default Profile**: Standard judge who assesses whether the question has been answered correctly
- **Implementation**:
  - Add profile selection in dashboard UI
  - Update config.py with profile system prompts
  - Pass selected profile to JudgeAgent initialization

## Medium Priority

### 4. Turn Faithfulness Metric
- **Description**: Measure what proportion of arguments are grounded in retrieved search context to prevent hallucination
- **Implementation**:
  - Compare argument content against retrieved search results
  - Calculate percentage of claims that can be traced to search context
  - Display faithfulness score per agent per round
  - Flag low faithfulness scores for review

### 5. Summary-Based Relay
- **Description**: Force agents to produce condensed summaries between rounds to reduce token bloat
- **Implementation**:
  - Add summary generation step at end of each round
  - Pass summaries instead of full arguments to next round
  - Store both full arguments and summaries for reference
  - Make summary length configurable

### 6. Search Efficiency Tracking
- **Description**: Track over-search rate (retrieving known info) vs. under-search rate (missing critical facts)
- **Implementation**:
  - Track when search is triggered vs. when it should have been triggered
  - Analyze search results to detect redundant retrievals
  - Calculate over-search and under-search rates
  - Display efficiency metrics in dashboard

### 7. Format Adherence and JSON Correctness
- **Description**: Track JSON correctness and ReAct pattern compliance (Thought-Action-Observation chains)
- **Implementation**:
  - Validate JSON structure of agent responses
  - Track ReAct pattern compliance (Thought → Action → Observation)
  - Log format errors and calculate success rate
  - Display parsing success rate per agent

### 8. Metrics Dashboard
- **Description**: Display all evaluation metrics in the UI
- **Implementation**:
  - Add new section in dashboard for evaluation metrics
  - Display real-time metrics during debate
  - Show historical metrics for completed debates
  - Add visualizations (charts, progress bars) for key metrics

## Implementation Order

1. **Phase 1**: Information Gain, Position Swapping, Judge Profiles (High Priority)
2. **Phase 2**: Turn Faithfulness, Summary-Based Relay (Medium Priority)
3. **Phase 3**: Search Efficiency, Format Adherence (Medium Priority)
4. **Phase 4**: Metrics Dashboard integration (Medium Priority)
