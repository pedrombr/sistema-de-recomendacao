### É necessário ter o uv instalado, caso não tenho:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### Depois instalar as depedências:
```bash
uv sync
```
### Para executar o projeto é só rodar:
```bash
uv run main.py
```
E fazer todos os imports por lá
