# Re-Chain
### This is a blockchain-based memory system for AI agents that solves context window limitations through persistent, retrievable interaction history.

##### Core Architecture

The system uses two agents working in tandem: an Active Agent (user-facing LLM like Claude) and a History Agent (blockchain-based memory layer). When the Active Agent exhausts its context window (around 500K tokens per block), the History Agent archives all interactions into blockchain blocks, creating an immutable, searchable record that persists across sessions.

##### The Memory Loop

Every interaction follows a continuous cycle: user prompts the Active Agent → History Agent logs the exchange → interactions accumulate until context limit → new block created in the chain → Active Agent retrieves relevant history when needed → loop continues. This creates unlimited conversational memory without degradation.

##### Intelligent Retrieval

After each user prompt, the Active Agent queries the blockchain API to access historical context. The system uses smart indexing (potentially via APIs from the LLM provider) to organize information for efficient retrieval with minimal token usage. When users reference past interactions, the system searches across blocks, checks adjacent blocks for extended tasks, and reconstructs full context even from fragmented conversations.

##### Integration Model

Implementation uses Clod's APIs to power the History Agent. The extension installs into existing LLMs (Claude, etc.) and automatically uploads interaction data every 500K tokens or at context boundaries. Everything routes through a Re-Chain API to a standalone app that manages the blockchain, handles subscriptions, and controls access.

##### Error Correction & Training Data

When users correct the Active Agent, the History Agent logs both the mistake and correction, creating a feedback loop. This accumulated data becomes training material to improve the Active Agent's performance over time, turning every error into a learning opportunity.

##### Subscription & Access Control

The system enforces limits at the app level, not the LLM level. If a user's subscription lapses and they run out of blocks, Claude continues uploading as normal, but the app blocks new data from being stored. This separates the AI interaction layer from the business logic layer cleanly.

##### Block Management

Blocks are 500K token chunks. Some interactions span multiple blocks when they exceed capacity. The History Agent tracks these relationships, ensuring the Active Agent can reconstruct split conversations by checking adjacent blocks for continuation.

##### Use Case

This solves the fundamental problem of AI amnesia—where conversations reset and context is lost. With blockchain persistence, the AI maintains perfect memory across unlimited interactions, learns from corrections, and retrieves exactly what's needed without reloading massive context windows.

##### Demo-ing Details

The main pipeline is runned through the .html file, currently the app is at a protoype 1 stage.
