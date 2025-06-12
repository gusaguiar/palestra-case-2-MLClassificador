# Sistema MLOps para Manutenção Preditiva Industrial

## Visão Geral

Este projeto implementa uma solução completa de MLOps para classificação de falhas em equipamentos industriais utilizando dados de sensores IoT. O sistema integra Apache Airflow para orquestração de pipelines, MLflow para gestão de experimentos e modelos, e APIs REST para inferência em tempo real.

## Dataset

O projeto utiliza o dataset AI4I 2020 de Manutenção Preditiva, focando especificamente em equipamentos do Tipo L, que apresenta 6.000 amostras e taxa de falhas de 3,92%.

### Features de Sensores
- **Temperatura do Ar [K]**: Temperatura ambiente do equipamento
- **Temperatura do Processo [K]**: Temperatura operacional durante produção
- **Velocidade Rotacional [rpm]**: Rotação da máquina em RPM
- **Torque [Nm]**: Torque aplicado durante operação
- **Desgaste da Ferramenta [min]**: Tempo acumulado de uso da ferramenta

### Target
- **Machine Failure**: Variável binária (0 = Normal, 1 = Falha)

## Engenharia de Features

O sistema implementa features derivadas baseadas em conhecimento de domínio industrial:

### Features Derivadas

1. **temp_difference**: `process_temperature - air_temperature` - Mede stress térmico
2. **estimated_power**: `torque * (rotational_speed * 2π / 60)` - Potência mecânica em Watts
3. **mechanical_stress**: `torque / (rotational_speed / 1000)` - Stress mecânico
4. **thermal_efficiency**: `process_temperature / air_temperature` - Eficiência térmica
5. **wear_per_operation**: `tool_wear / (rotational_speed * 0.001)` - Desgaste normalizado

## Arquitetura MLOps

### DAGs do Airflow

#### 1. Feature Engineering (`feature_engineering_pipeline`)
- **Frequência**: Diária
- **Função**: Processa dados brutos e extrai features
- **Tarefas**:
  - Validação de qualidade dos dados AI4I 2020 Type L
  - Extração de features de sensores
  - Validação de features derivadas
  - Limpeza de arquivos antigos

#### 2. Treinamento de Modelo (`model_training_pipeline`)
- **Frequência**: Semanal
- **Função**: Treina modelos com MLflow
- **Tarefas**:
  - Validação de dados de treinamento
  - Treinamento de Random Forest e SVM
  - Avaliação e comparação de modelos
  - Registro automático no MLflow Model Registry
  - Promoção inteligente baseada em F1-Score

#### 3. Inferência em Lote (`batch_inference_pipeline`)
- **Frequência**: Diária
- **Função**: Predições em novos dados
- **Tarefas**:
  - Carregamento do modelo de produção
  - Processamento de novos dados
  - Geração de predições
  - Relatório de inferência

## Configuração e Execução

### Pré-requisitos

**Docker**: Obrigatório para execução do projeto.

### Inicialização

```powershell
# Clone e inicialize
git clone <repository-url>
cd palestra-case-2-MLClassificador
docker-compose up -d

# Verificar status
docker-compose ps
```

### Interfaces
- **Airflow**: http://localhost:8080 (admin/admin)
- **MLflow**: http://localhost:5000
- **API**: http://localhost:8000/docs

### Estrutura de Diretórios
```
data/
├── raw/                    # Dados originais
├── processed/              # Features processadas
└── inference/              # Dados para predição

src/
├── features/               # Pipeline de features
├── models/                 # Treinamento
└── api/                    # APIs REST

pipelines/airflow/          # DAGs
```

## APIs de Inferência

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature": 298.5,
    "process_temperature": 308.9,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 108
  }'
```

**Resposta**:
```json
{
  "prediction": 0,
  "probability": 0.92,
  "risk_level": "Baixo",
  "recommendation": "Operação normal. Continuar monitoramento."
}
```

## Algoritmos Utilizados

- **Random Forest**: Ensemble robusto com boa interpretabilidade
- **SVM**: Support Vector Machine para classificação

## MLflow Integration

### Model Registry
- **Registro Automático**: Todos os modelos são registrados
- **Versionamento**: Controle completo de versões
- **Promoção Inteligente**: Baseada em performance superior
- **Rollback**: Downgrade automático em caso de degradação

### Experiment Tracking
- **Hiperparâmetros**: Logging completo
- **Métricas**: Acurácia, Precision, Recall, F1-Score, ROC-AUC
- **Artefatos**: Gráficos, matrizes de confusão, feature importance
- **Reproducibilidade**: Seeds fixos e versionamento

## Interpretação de Resultados

- **Predição 0 (Normal)**: Continuar operação normal
- **Predição 1 (Falha)**: Parar para inspeção imediata

## Troubleshooting

### Problemas Comuns

```powershell
# Verificar logs
docker-compose logs [serviço]

# Rebuild completo
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Reiniciar Airflow
docker-compose restart airflow-webserver airflow-scheduler airflow-worker
```

## Tecnologias Utilizadas

### Infraestrutura
- **Docker & Docker Compose**: Containerização
- **Apache Airflow**: Orquestração de pipelines
- **PostgreSQL**: Metastore
- **Redis**: Message broker

### Machine Learning
- **MLflow**: Gestão de experimentos
- **Scikit-learn**: Algoritmos de ML
- **Pandas & NumPy**: Manipulação de dados

### APIs
- **FastAPI**: Framework web
- **Uvicorn**: Servidor ASGI