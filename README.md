# Brain Dex (TRIBE v2 + UV)

Projeto em Python usando UV para prever resposta cerebral (fMRI in-silico) a partir de videos com o modelo [facebook/tribev2](https://huggingface.co/facebook/tribev2).

## O que este projeto faz

- Carrega o modelo TRIBE v2.
- Processa um video (`.mp4`).
- Gera predicoes de atividade cerebral ao longo do tempo.
- Salva:
	- `outputs/summary.json` (metricas resumidas)
	- `outputs/events.csv` (eventos extraidos do video)
	- `outputs/segments.csv` (segmentacao temporal da inferencia)
	- `outputs/predictions.npy` (opcional, matriz completa)

## 1) Preparacao do ambiente

### Requisitos de software

- Windows 10/11, Linux ou macOS
- [UV](https://docs.astral.sh/uv/) instalado
- Python 3.11 ou 3.12
- Git instalado

### Criar/instalar dependencias

No diretorio do projeto:

```powershell
uv sync
```

Isso instala todas as dependencias declaradas no `pyproject.toml`, incluindo o `tribev2` via repositório oficial.

## 2) Modelos e downloads necessarios

### Modelo principal

- O projeto usa `TribeModel.from_pretrained("facebook/tribev2")`.
- Os pesos e artefatos sao baixados automaticamente na primeira execucao.

### Onde os arquivos ficam

- Por padrao, este projeto usa `./cache` (parametro `--cache-dir`).

### Preciso baixar algo manualmente?

- Para **uso com video**, normalmente **nao**: o download e automatico.
- Se voce quiser usar entradas de texto com alguns encoders especificos, pode haver necessidade de autenticacao no Hugging Face para modelos gated (conforme documentacao oficial do TRIBE v2).

## 3) Executar inferencia em video

Exemplo:

```powershell
uv run brain-dex --video "C:\caminho\para\meu_video.mp4"
```

### Escolha de dispositivo

O comando aceita `--device`:

- `auto`: usa GPU NVIDIA se existir, depois MPS, e por fim CPU
- `cuda`: força GPU NVIDIA/CUDA
- `cpu`: força CPU
- `mps`: Apple Silicon, se aplicavel
- `openvino`: atualmente nao suportado pelo TRIBE v2, e o comando encerra com mensagem clara

Exemplo para sua RTX 3060 Ti:

```powershell
uv run brain-dex --video "C:\caminho\para\meu_video.mp4" --device cuda
```

Com opcoes:

```powershell
uv run brain-dex \
	--video "C:\caminho\para\meu_video.mp4" \
	--cache-dir ".\cache" \
	--output-dir ".\outputs" \
	--save-full-preds
```

## 4) Como interpretar o resultado

O `summary.json` traz estatisticas globais da predicao:

- `timesteps`: numero de janelas temporais analisadas
- `vertices`: numero de vertices cerebrais previstos (malha cortical)
- `mean`, `std`, `min`, `max`, `p05`, `p50`, `p95`, `mean_abs`: resumo estatistico da atividade prevista

Esses valores representam a intensidade prevista de resposta neural no espaco da malha usada pelo modelo.

## 5) Requisitos minimos recomendados (rodar com leveza)

### Minimo funcional (CPU)

- CPU: 6+ nucleos modernos
- RAM: 16 GB
- Disco livre: 25 GB (dependencias + cache de modelos)
- GPU: nao obrigatoria, mas execucao pode ficar lenta

### Recomendado (fluidez)

- CPU: 8+ nucleos
- RAM: 32 GB
- GPU NVIDIA com CUDA e 8 GB+ VRAM
- Disco SSD com 40 GB+ livres

### Melhor experiencia na RTX 3060 Ti

- Use `--device cuda`
- Mantenha os drivers NVIDIA atualizados
- Instale uma build do PyTorch com CUDA, se desejar acelerar mais do que a versao CPU atual

### Sobre Intel Arc / OpenVINO

- O pacote TRIBE v2 usado neste projeto nao expõe backend OpenVINO.
- Para nao quebrar o projeto, mantivemos CPU/CUDA/MPS via `device=` e retornamos erro explicito em `--device openvino`.
- Se quiser usar Intel Arc com aceleração real, seria necessario portar/exportar o modelo para ONNX/OpenVINO, o que e um trabalho separado e pode exigir ajustes no modelo.

## 6) Dicas de performance

- Use videos curtos para teste inicial (10s a 30s).
- Mantenha cache local (`--cache-dir`) para evitar re-download.
- Feche apps pesados ao rodar inferencia em CPU.
- Se tiver GPU CUDA configurada no PyTorch, a inferencia tende a ser muito mais rapida.

## 7) Solucao de problemas

- Erro de dependencias: execute `uv lock --upgrade` e depois `uv sync`.
- Falha de download de modelo: verifique internet/firewall e tente novamente.
- Memoria insuficiente: use videos menores e feche outros processos.
- Se a GPU NVIDIA nao aparecer, confirme se voce instalou uma build do PyTorch com CUDA e se `torch.cuda.is_available()` retorna `True`.

## 8) Observacao sobre a fonte do modelo

A pagina do Hugging Face (`facebook/tribev2`) referencia os pesos/documentacao. O pacote Python usado aqui e instalado a partir do repositório oficial de codigo (`facebookresearch/tribev2`) para garantir instalacao correta via UV.
