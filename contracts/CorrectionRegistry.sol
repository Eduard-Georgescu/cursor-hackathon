// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title CorrectionRegistry
/// @notice Public, append-only registry of MCP Agent System correction and
///         context-block proofs. Built for the Blockchain for Good Alliance,
///         supporting SDG 4 (Quality Education) and SDG 16 (Strong Institutions)
///         by making AI training data corrections auditable on-chain.
contract CorrectionRegistry {
    event CorrectionRecorded(
        bytes16 indexed sessionId,
        bytes32 originalHash,
        bytes32 correctionHash,
        uint256 timestamp
    );

    event ContextBlockRecorded(
        bytes16 indexed sessionId,
        bytes16 blockId,
        bytes32 summaryHash,
        uint256 timestamp
    );

    /// @notice Record a hash-only proof of a correction. Raw text is never
    ///         stored on chain to preserve user privacy; only SHA-256 hashes.
    function recordCorrection(
        bytes16 sessionId,
        bytes32 originalHash,
        bytes32 correctionHash
    ) external {
        emit CorrectionRecorded(sessionId, originalHash, correctionHash, block.timestamp);
    }

    /// @notice Record that a context block was archived. The hash references
    ///         the LLM-generated summary so reviewers can verify integrity.
    function recordContextBlock(
        bytes16 sessionId,
        bytes16 blockId,
        bytes32 summaryHash
    ) external {
        emit ContextBlockRecorded(sessionId, blockId, summaryHash, block.timestamp);
    }
}
