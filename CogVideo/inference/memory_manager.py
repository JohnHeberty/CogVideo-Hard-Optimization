"""
Memory Manager - Controle inteligente de RAM e VRAM

Este módulo gerencia automaticamente a memória RAM e VRAM,
descarregando modelos após o uso para liberar recursos para outros microserviços.
"""

import gc
import torch
import psutil
import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Gerenciador de memória que controla RAM e VRAM automaticamente.
    
    Features:
    - Descarrega modelos após uso
    - Limpa cache CUDA agressivamente
    - Monitora uso de memória
    - Libera recursos para outros serviços
    """
    
    def __init__(self, aggressive_cleanup: bool = True):
        """
        Args:
            aggressive_cleanup: Se True, faz limpeza agressiva de memória após cada operação
        """
        self.aggressive_cleanup = aggressive_cleanup
        self.loaded_models: Dict[str, Any] = {}
        self.model_loaders: Dict[str, Callable] = {}
        
    def register_model_loader(self, name: str, loader: Callable):
        """
        Registra uma função que carrega um modelo.
        
        Args:
            name: Nome identificador do modelo
            loader: Função que retorna o modelo carregado
        """
        self.model_loaders[name] = loader
        logger.info(f"✅ Model loader registered: {name}")
        
    @contextmanager
    def load_model(self, name: str):
        """
        Context manager para carregar modelo temporariamente.
        Garante que o modelo será descarregado após o uso.
        
        Usage:
            with memory_manager.load_model("upscale") as model:
                result = model.process(data)
            # Modelo automaticamente descarregado aqui
        
        Args:
            name: Nome do modelo registrado
            
        Yields:
            O modelo carregado
        """
        model = None
        try:
            # Carrega modelo se não estiver em cache
            if name not in self.loaded_models:
                logger.info(f"📥 Loading model: {name}")
                self._log_memory_before(name)
                
                if name not in self.model_loaders:
                    raise ValueError(f"Model loader not registered: {name}")
                
                model = self.model_loaders[name]()
                self.loaded_models[name] = model
                
                self._log_memory_after(name)
            else:
                logger.info(f"♻️ Reusing cached model: {name}")
                model = self.loaded_models[name]
            
            yield model
            
        finally:
            # Sempre descarrega após uso (se aggressive_cleanup ativado)
            if self.aggressive_cleanup and name in self.loaded_models:
                self.unload_model(name)
                
    def unload_model(self, name: str):
        """
        Descarrega um modelo específico da memória.
        
        Args:
            name: Nome do modelo a descarregar
        """
        if name in self.loaded_models:
            logger.info(f"🗑️ Unloading model: {name}")
            
            try:
                del self.loaded_models[name]
                self._cleanup_memory()
                logger.info(f"✅ Model unloaded: {name}")
            except Exception as e:
                logger.error(f"❌ Error unloading model {name}: {e}")
                
    def unload_all_models(self):
        """Descarrega todos os modelos carregados."""
        logger.info("🗑️ Unloading all models...")
        
        model_names = list(self.loaded_models.keys())
        for name in model_names:
            self.unload_model(name)
            
        logger.info("✅ All models unloaded")
        
    def _cleanup_memory(self):
        """Limpeza agressiva de memória RAM e VRAM."""
        # Garbage collection Python
        gc.collect()
        
        # Limpeza CUDA se disponível
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Limpeza mais agressiva
            try:
                torch.cuda.ipc_collect()
            except:
                pass
                
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas de uso de memória.
        
        Returns:
            Dict com uso de RAM e VRAM em GB
        """
        stats = {}
        
        # RAM
        ram = psutil.virtual_memory()
        stats['ram_used_gb'] = ram.used / (1024**3)
        stats['ram_total_gb'] = ram.total / (1024**3)
        stats['ram_percent'] = ram.percent
        
        # VRAM
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / (1024**3)
            vram_reserved = torch.cuda.memory_reserved() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            stats['vram_allocated_gb'] = vram_allocated
            stats['vram_reserved_gb'] = vram_reserved
            stats['vram_total_gb'] = vram_total
            stats['vram_percent'] = (vram_reserved / vram_total) * 100
        else:
            stats['vram_allocated_gb'] = 0
            stats['vram_reserved_gb'] = 0
            stats['vram_total_gb'] = 0
            stats['vram_percent'] = 0
            
        return stats
        
    def _log_memory_before(self, operation: str):
        """Log de memória antes de uma operação."""
        stats = self.get_memory_stats()
        logger.info(
            f"📊 Memory before {operation}: "
            f"RAM {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f}GB ({stats['ram_percent']:.1f}%) | "
            f"VRAM {stats['vram_allocated_gb']:.1f}/{stats['vram_total_gb']:.1f}GB ({stats['vram_percent']:.1f}%)"
        )
        
    def _log_memory_after(self, operation: str):
        """Log de memória depois de uma operação."""
        stats = self.get_memory_stats()
        logger.info(
            f"📊 Memory after {operation}: "
            f"RAM {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f}GB ({stats['ram_percent']:.1f}%) | "
            f"VRAM {stats['vram_allocated_gb']:.1f}/{stats['vram_total_gb']:.1f}GB ({stats['vram_percent']:.1f}%)"
        )
        
    def force_cleanup(self):
        """
        Força limpeza completa de memória.
        Descarrega todos os modelos e limpa caches.
        """
        logger.warning("⚠️ FORCING COMPLETE MEMORY CLEANUP...")
        
        self._log_memory_before("force_cleanup")
        
        # Descarrega todos os modelos
        self.unload_all_models()
        
        # Limpeza agressiva múltiplas vezes
        for _ in range(3):
            self._cleanup_memory()
            
        self._log_memory_after("force_cleanup")
        
    @contextmanager
    def temporary_operation(self, operation_name: str):
        """
        Context manager para operações temporárias.
        Garante limpeza de memória ao final.
        
        Usage:
            with memory_manager.temporary_operation("video_generation"):
                # Sua operação aqui
                generate_video()
            # Memória automaticamente limpa aqui
        
        Args:
            operation_name: Nome da operação para logging
        """
        logger.info(f"🚀 Starting operation: {operation_name}")
        self._log_memory_before(operation_name)
        
        try:
            yield
        finally:
            logger.info(f"🧹 Cleaning up after: {operation_name}")
            
            if self.aggressive_cleanup:
                self._cleanup_memory()
                
            self._log_memory_after(operation_name)
            
    def check_memory_available(self, required_vram_gb: float = 10.0) -> bool:
        """
        Verifica se há memória VRAM suficiente disponível.
        
        Args:
            required_vram_gb: VRAM necessária em GB
            
        Returns:
            True se há memória suficiente
        """
        if not torch.cuda.is_available():
            return False
            
        stats = self.get_memory_stats()
        available_vram = stats['vram_total_gb'] - stats['vram_allocated_gb']
        
        if available_vram < required_vram_gb:
            logger.warning(
                f"⚠️ Insufficient VRAM: {available_vram:.1f}GB available, "
                f"{required_vram_gb:.1f}GB required"
            )
            return False
            
        return True
        
    def auto_cleanup_if_needed(self, threshold_percent: float = 80.0):
        """
        Limpa memória automaticamente se uso ultrapassar threshold.
        
        Args:
            threshold_percent: Percentual de uso que dispara limpeza
        """
        stats = self.get_memory_stats()
        
        # Verifica RAM
        if stats['ram_percent'] > threshold_percent:
            logger.warning(
                f"⚠️ RAM usage high ({stats['ram_percent']:.1f}%), "
                f"triggering cleanup..."
            )
            self.force_cleanup()
            
        # Verifica VRAM
        if stats['vram_percent'] > threshold_percent:
            logger.warning(
                f"⚠️ VRAM usage high ({stats['vram_percent']:.1f}%), "
                f"triggering cleanup..."
            )
            self.force_cleanup()


# Singleton global
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(aggressive_cleanup: bool = True) -> MemoryManager:
    """
    Retorna instância singleton do gerenciador de memória.
    
    Args:
        aggressive_cleanup: Ativa limpeza agressiva após operações
        
    Returns:
        Instância do MemoryManager
    """
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(aggressive_cleanup=aggressive_cleanup)
        logger.info("✅ Memory Manager initialized")
        
    return _global_memory_manager
