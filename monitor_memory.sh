#!/bin/bash
# Monitor de memória em tempo real

echo "🔍 Memory Monitor - Pressione Ctrl+C para parar"
echo "=============================================="

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🧠 MEMORY MONITOR - $(date +%H:%M:%S)                     ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Memory stats do container
    docker exec cogvideo python3 -c '
import sys
sys.path.insert(0, "/workspace/CogVideo/inference")
from memory_manager import get_memory_manager
mm = get_memory_manager()
stats = mm.get_memory_stats()

print("📊 MEMORY USAGE:")
print(f"  RAM:  {stats[\"ram_used_gb\"]:.2f}GB / {stats[\"ram_total_gb\"]:.2f}GB ({stats[\"ram_percent\"]:.1f}%)")
print(f"  VRAM: {stats[\"vram_allocated_gb\"]:.2f}GB / {stats[\"vram_total_gb\"]:.2f}GB ({stats[\"vram_percent\"]:.1f}%)")
print()
print("🔧 LOADED MODELS:")
if mm.loaded_models:
    for name in mm.loaded_models.keys():
        print(f"  • {name}")
else:
    print("  ✅ No models loaded (idle state)")
' 2>/dev/null || echo "⚠️ Container não está respondendo"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🖥️  HOST SYSTEM:"
    free -h | grep -E "Mem:|Swap:"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 3
done
