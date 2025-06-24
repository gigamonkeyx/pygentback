"""
Transaction Coordinator

Manages atomic operations across multiple MCP servers with ACID properties,
distributed transaction support, and automatic rollback capabilities.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .coordination_models import (
    MCPServerInfo, OrchestrationConfig, ServerID, TaskID
)
from .mcp_orchestrator import MCPOrchestrator

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction states."""
    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class TransactionOperation:
    """Individual operation within a transaction."""
    operation_id: str
    server_id: str
    operation_type: str
    operation_data: Dict[str, Any]
    compensation_data: Optional[Dict[str, Any]] = None
    state: TransactionState = TransactionState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DistributedTransaction:
    """Distributed transaction across multiple MCP servers."""
    transaction_id: str
    operations: List[TransactionOperation] = field(default_factory=list)
    state: TransactionState = TransactionState.PENDING
    coordinator_id: str = ""
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if transaction has expired."""
        return datetime.utcnow() - self.created_at > self.timeout
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get transaction duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class TransactionCoordinator:
    """
    Distributed transaction coordinator for multi-MCP operations.
    
    Features:
    - Two-phase commit protocol (2PC)
    - Automatic rollback on failures
    - Timeout handling and recovery
    - Saga pattern for long-running transactions
    - Compensation-based error recovery
    """
    
    def __init__(self, config: OrchestrationConfig, mcp_orchestrator: MCPOrchestrator):
        self.config = config
        self.mcp_orchestrator = mcp_orchestrator
        
        # Transaction management
        self.active_transactions: Dict[str, DistributedTransaction] = {}
        self.completed_transactions: Dict[str, DistributedTransaction] = {}
        self.transaction_history: List[DistributedTransaction] = []
        
        # Coordination state
        self.coordinator_id = str(uuid.uuid4())
        self.is_running = False
        
        # Recovery and monitoring
        self.recovery_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.transaction_stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'aborted_transactions': 0,
            'avg_duration': 0.0
        }
        
        logger.info(f"Transaction Coordinator initialized: {self.coordinator_id}")
    
    async def start(self):
        """Start the transaction coordinator."""
        self.is_running = True
        
        # Start background tasks
        self.recovery_task = asyncio.create_task(self._recovery_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Transaction Coordinator started")
    
    async def stop(self):
        """Stop the transaction coordinator."""
        self.is_running = False
        
        # Cancel background tasks
        if self.recovery_task:
            self.recovery_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Abort all active transactions
        for transaction in list(self.active_transactions.values()):
            await self.abort_transaction(transaction.transaction_id, "Coordinator shutdown")
        
        logger.info("Transaction Coordinator stopped")
    
    async def begin_transaction(self, 
                              operations: List[Dict[str, Any]],
                              timeout: timedelta = None,
                              metadata: Dict[str, Any] = None) -> str:
        """Begin a new distributed transaction."""
        try:
            transaction_id = str(uuid.uuid4())
            timeout = timeout or timedelta(minutes=5)
            metadata = metadata or {}
            
            # Create transaction operations
            transaction_ops = []
            for i, op_data in enumerate(operations):
                operation = TransactionOperation(
                    operation_id=f"{transaction_id}_{i}",
                    server_id=op_data['server_id'],
                    operation_type=op_data['operation_type'],
                    operation_data=op_data['operation_data'],
                    compensation_data=op_data.get('compensation_data')
                )
                transaction_ops.append(operation)
            
            # Create distributed transaction
            transaction = DistributedTransaction(
                transaction_id=transaction_id,
                operations=transaction_ops,
                coordinator_id=self.coordinator_id,
                timeout=timeout,
                metadata=metadata
            )
            
            # Register transaction
            self.active_transactions[transaction_id] = transaction
            self.transaction_stats['total_transactions'] += 1
            
            logger.info(f"Started transaction {transaction_id} with {len(operations)} operations")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Failed to begin transaction: {e}")
            raise
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a distributed transaction using two-phase commit."""
        try:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            if transaction.is_expired:
                await self.abort_transaction(transaction_id, "Transaction expired")
                return False
            
            transaction.state = TransactionState.PREPARING
            transaction.started_at = datetime.utcnow()
            
            logger.info(f"Committing transaction {transaction_id}")
            
            # Phase 1: Prepare all operations
            prepare_success = await self._prepare_phase(transaction)
            
            if not prepare_success:
                await self.abort_transaction(transaction_id, "Prepare phase failed")
                return False
            
            # Phase 2: Commit all operations
            commit_success = await self._commit_phase(transaction)
            
            if commit_success:
                transaction.state = TransactionState.COMMITTED
                transaction.completed_at = datetime.utcnow()
                
                # Move to completed transactions
                self.completed_transactions[transaction_id] = transaction
                del self.active_transactions[transaction_id]
                
                # Update statistics
                self.transaction_stats['successful_transactions'] += 1
                self._update_avg_duration(transaction)
                
                logger.info(f"Transaction {transaction_id} committed successfully")
                return True
            else:
                await self.abort_transaction(transaction_id, "Commit phase failed")
                return False
                
        except Exception as e:
            logger.error(f"Transaction commit failed for {transaction_id}: {e}")
            await self.abort_transaction(transaction_id, f"Commit error: {str(e)}")
            return False
    
    async def abort_transaction(self, transaction_id: str, reason: str = "Manual abort") -> bool:
        """Abort a distributed transaction with compensation."""
        try:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                logger.warning(f"Transaction {transaction_id} not found for abort")
                return False
            
            transaction.state = TransactionState.ABORTING
            
            logger.info(f"Aborting transaction {transaction_id}: {reason}")
            
            # Execute compensation operations for completed operations
            compensation_success = await self._compensate_operations(transaction)
            
            transaction.state = TransactionState.ABORTED
            transaction.completed_at = datetime.utcnow()
            transaction.metadata['abort_reason'] = reason
            
            # Move to completed transactions
            self.completed_transactions[transaction_id] = transaction
            del self.active_transactions[transaction_id]
            
            # Update statistics
            self.transaction_stats['aborted_transactions'] += 1
            
            logger.info(f"Transaction {transaction_id} aborted")
            return compensation_success
            
        except Exception as e:
            logger.error(f"Transaction abort failed for {transaction_id}: {e}")
            return False
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a transaction."""
        # Check active transactions
        if transaction_id in self.active_transactions:
            transaction = self.active_transactions[transaction_id]
        elif transaction_id in self.completed_transactions:
            transaction = self.completed_transactions[transaction_id]
        else:
            return None
        
        return {
            'transaction_id': transaction.transaction_id,
            'state': transaction.state.value,
            'operations_count': len(transaction.operations),
            'operations': [
                {
                    'operation_id': op.operation_id,
                    'server_id': op.server_id,
                    'operation_type': op.operation_type,
                    'state': op.state.value,
                    'error': op.error
                }
                for op in transaction.operations
            ],
            'created_at': transaction.created_at.isoformat(),
            'started_at': transaction.started_at.isoformat() if transaction.started_at else None,
            'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None,
            'duration': transaction.duration.total_seconds() if transaction.duration else None,
            'is_expired': transaction.is_expired,
            'metadata': transaction.metadata
        }
    
    async def _prepare_phase(self, transaction: DistributedTransaction) -> bool:
        """Execute prepare phase of two-phase commit."""
        try:
            prepare_tasks = []
            
            for operation in transaction.operations:
                task = asyncio.create_task(self._prepare_operation(operation))
                prepare_tasks.append(task)
            
            # Wait for all prepare operations
            results = await asyncio.gather(*prepare_tasks, return_exceptions=True)
            
            # Check if all operations prepared successfully
            all_prepared = True
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    transaction.operations[i].state = TransactionState.FAILED
                    transaction.operations[i].error = str(result)
                    all_prepared = False
                elif not result:
                    all_prepared = False
            
            if all_prepared:
                transaction.state = TransactionState.PREPARED
                logger.debug(f"Transaction {transaction.transaction_id} prepared successfully")
            else:
                logger.warning(f"Transaction {transaction.transaction_id} prepare phase failed")
            
            return all_prepared
            
        except Exception as e:
            logger.error(f"Prepare phase failed for transaction {transaction.transaction_id}: {e}")
            return False
    
    async def _prepare_operation(self, operation: TransactionOperation) -> bool:
        """Prepare a single operation."""
        try:
            operation.state = TransactionState.PREPARING
            
            # Create prepare request
            prepare_request = {
                'operation': 'prepare',
                'transaction_id': operation.operation_id,
                'operation_type': operation.operation_type,
                'operation_data': operation.operation_data
            }
            
            # Execute prepare on MCP server
            result = await self.mcp_orchestrator.execute_request(
                capability=operation.operation_type,
                request=prepare_request,
                preferred_server=operation.server_id
            )
            
            if result.get('status') == 'success':
                operation.state = TransactionState.PREPARED
                operation.result = result
                return True
            else:
                operation.state = TransactionState.FAILED
                operation.error = result.get('error', 'Prepare failed')
                return False
                
        except Exception as e:
            operation.state = TransactionState.FAILED
            operation.error = str(e)
            logger.error(f"Operation prepare failed {operation.operation_id}: {e}")
            return False
    
    async def _commit_phase(self, transaction: DistributedTransaction) -> bool:
        """Execute commit phase of two-phase commit."""
        try:
            transaction.state = TransactionState.COMMITTING
            commit_tasks = []
            
            for operation in transaction.operations:
                if operation.state == TransactionState.PREPARED:
                    task = asyncio.create_task(self._commit_operation(operation))
                    commit_tasks.append(task)
            
            # Wait for all commit operations
            results = await asyncio.gather(*commit_tasks, return_exceptions=True)
            
            # Check if all operations committed successfully
            all_committed = True
            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    all_committed = False
                    break
            
            return all_committed
            
        except Exception as e:
            logger.error(f"Commit phase failed for transaction {transaction.transaction_id}: {e}")
            return False
    
    async def _commit_operation(self, operation: TransactionOperation) -> bool:
        """Commit a single operation."""
        try:
            operation.state = TransactionState.COMMITTING
            
            # Create commit request
            commit_request = {
                'operation': 'commit',
                'transaction_id': operation.operation_id,
                'operation_type': operation.operation_type,
                'operation_data': operation.operation_data
            }
            
            # Execute commit on MCP server
            result = await self.mcp_orchestrator.execute_request(
                capability=operation.operation_type,
                request=commit_request,
                preferred_server=operation.server_id
            )
            
            if result.get('status') == 'success':
                operation.state = TransactionState.COMMITTED
                operation.result = result
                return True
            else:
                operation.state = TransactionState.FAILED
                operation.error = result.get('error', 'Commit failed')
                return False
                
        except Exception as e:
            operation.state = TransactionState.FAILED
            operation.error = str(e)
            logger.error(f"Operation commit failed {operation.operation_id}: {e}")
            return False
    
    async def _compensate_operations(self, transaction: DistributedTransaction) -> bool:
        """Execute compensation operations for rollback."""
        try:
            compensation_tasks = []
            
            # Compensate operations in reverse order
            for operation in reversed(transaction.operations):
                if operation.state in [TransactionState.COMMITTED, TransactionState.PREPARED]:
                    if operation.compensation_data:
                        task = asyncio.create_task(self._compensate_operation(operation))
                        compensation_tasks.append(task)
            
            if compensation_tasks:
                # Wait for all compensation operations
                results = await asyncio.gather(*compensation_tasks, return_exceptions=True)
                
                # Check compensation results
                compensation_success = all(
                    not isinstance(result, Exception) and result
                    for result in results
                )
                
                return compensation_success
            
            return True  # No compensation needed
            
        except Exception as e:
            logger.error(f"Compensation failed for transaction {transaction.transaction_id}: {e}")
            return False
    
    async def _compensate_operation(self, operation: TransactionOperation) -> bool:
        """Execute compensation for a single operation."""
        try:
            if not operation.compensation_data:
                return True
            
            # Create compensation request
            compensation_request = {
                'operation': 'compensate',
                'transaction_id': operation.operation_id,
                'operation_type': operation.operation_type,
                'compensation_data': operation.compensation_data
            }
            
            # Execute compensation on MCP server
            result = await self.mcp_orchestrator.execute_request(
                capability=operation.operation_type,
                request=compensation_request,
                preferred_server=operation.server_id
            )
            
            return result.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"Operation compensation failed {operation.operation_id}: {e}")
            return False
    
    async def _recovery_loop(self):
        """Background loop for transaction recovery."""
        while self.is_running:
            try:
                # Check for expired transactions
                expired_transactions = [
                    tx_id for tx_id, tx in self.active_transactions.items()
                    if tx.is_expired
                ]
                
                # Abort expired transactions
                for tx_id in expired_transactions:
                    await self.abort_transaction(tx_id, "Transaction timeout")
                
                # Check for stuck transactions
                stuck_transactions = [
                    tx_id for tx_id, tx in self.active_transactions.items()
                    if tx.state == TransactionState.PREPARING and
                    datetime.utcnow() - tx.created_at > timedelta(minutes=2)
                ]
                
                # Abort stuck transactions
                for tx_id in stuck_transactions:
                    await self.abort_transaction(tx_id, "Transaction stuck in prepare phase")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up old transactions."""
        while self.is_running:
            try:
                # Clean up old completed transactions
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                old_transactions = [
                    tx_id for tx_id, tx in self.completed_transactions.items()
                    if tx.completed_at and tx.completed_at < cutoff_time
                ]
                
                for tx_id in old_transactions:
                    # Move to history before deletion
                    self.transaction_history.append(self.completed_transactions[tx_id])
                    del self.completed_transactions[tx_id]
                
                # Limit history size
                if len(self.transaction_history) > 1000:
                    self.transaction_history = self.transaction_history[-1000:]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    def _update_avg_duration(self, transaction: DistributedTransaction):
        """Update average transaction duration."""
        if transaction.duration:
            current_avg = self.transaction_stats['avg_duration']
            total_successful = self.transaction_stats['successful_transactions']
            
            # Calculate new average
            new_avg = (
                (current_avg * (total_successful - 1) + transaction.duration.total_seconds()) /
                total_successful
            )
            self.transaction_stats['avg_duration'] = new_avg
    
    async def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator metrics."""
        return {
            'coordinator_id': self.coordinator_id,
            'active_transactions': len(self.active_transactions),
            'completed_transactions': len(self.completed_transactions),
            'transaction_history_size': len(self.transaction_history),
            'statistics': dict(self.transaction_stats),
            'is_running': self.is_running
        }