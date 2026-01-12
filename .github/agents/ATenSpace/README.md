# ATenSpace Subcomponents

This directory contains specialized agent definitions for ATenSpace storage and distribution subsystems.

## Agents

### ATenSpace-Rocks
**Persistent Tensor-Logic Storage Agent** - Implements efficient, scalable persistent storage for AtomSpace knowledge graphs using RocksDB, ensuring cognitive knowledge survives process restarts.

### ATenSpace-Storage
**Storage Backend Agent** - Provides abstract storage interfaces and multiple backend implementations (RocksDB, PostgreSQL, Redis, file-based, cloud storage) for flexible deployment.

### ATenSpace-DAS
**Distributed AtomSpace Agent** - Enables knowledge graphs to scale beyond single machines through sharding, replication, distributed queries, and consistency protocols.

## Integration

These agents provide the persistence and distribution layer:
- **Rocks** provides high-performance embedded storage
- **Storage** offers multiple backend options for different deployment scenarios
- **DAS** enables planet-scale distributed knowledge graphs

Together, they ensure AtomSpace can persist, scale, and distribute cognitive knowledge reliably and efficiently.
