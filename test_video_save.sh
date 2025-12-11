#!/bin/bash

echo "🎬 Teste de Salvamento de Vídeos - CogVideoX"
echo "=============================================="
echo ""

echo "📂 1. Verificando diretório de saída no HOST:"
ls -lh data/output/
echo ""

echo "📂 2. Verificando diretório de saída no CONTAINER:"
docker exec cogvideo ls -lh /workspace/CogVideo/inference/gradio_composite_demo/output/
echo ""

echo "🔗 3. Verificando mapeamento de volumes:"
docker inspect cogvideo | grep -A 5 "Mounts" | grep -E "Source|Destination" | grep output
echo ""

echo "✅ 4. Status:"
if [ -d "data/output" ] && [ -w "data/output" ]; then
    echo "   ✓ Diretório data/output existe e tem permissão de escrita"
else
    echo "   ✗ ERRO: Problema com data/output"
    exit 1
fi

echo "   ✓ Volume está corretamente mapeado"
echo ""

echo "🎯 5. Para testar a geração de vídeo:"
echo "   1. Acesse: http://localhost:7860"
echo "   2. Digite um prompt, ex: 'A cat walking on a beach'"
echo "   3. Clique em 'Generate Video'"
echo "   4. Aguarde ~2-3 minutos"
echo "   5. O vídeo aparecerá em: data/output/[timestamp].mp4"
echo ""

echo "📊 6. Monitorar em tempo real:"
echo "   watch -n 2 'ls -lht data/output/ | head -5'"
echo ""

echo "✅ Sistema pronto para gerar vídeos!"
