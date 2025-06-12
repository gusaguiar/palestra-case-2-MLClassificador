# 📁 Diretório de Features

Este diretório contém as features em diferentes estágios de processamento.

## 📊 Estrutura Atual

```
data/
├── processed/
│   └── features_raw.csv     # Features extraídas mas NÃO normalizadas
└── features/
    └── (vazio)              # Reservado para features intermediárias
```

## 🔄 Fluxo de Processamento

### 1. Raw Data → Features Extraídas
- **Entrada**: `data/raw/ai4i2020.csv`
- **Saída**: `data/processed/features_raw.csv`
- **Processo**: Extração de 10 features (5 originais + 5 derivadas)

### 2. Features Extraídas → Features Normalizadas
- **Entrada**: `data/processed/features_raw.csv`
- **Saída**: Arrays normalizados durante treinamento
- **Processo**: Normalização aplicada APÓS train/test split

## 🚫 Por que `/features` está Vazio?

A pasta `/features` está vazia **intencionalmente** para evitar **data leakage**:

1. **Features Extraídas**: Salvas em `/processed/features_raw.csv`
2. **Normalização**: Aplicada apenas durante treinamento
3. **Sem Persistência**: Dados normalizados não são salvos

## 📋 Features Disponíveis

### Originais (Sensores IoT):
- `air_temperature` - Temperatura do Ar [K]
- `process_temperature` - Temperatura do Processo [K]
- `rotational_speed` - Velocidade Rotacional [rpm]
- `torque` - Torque [Nm]
- `tool_wear` - Desgaste da Ferramenta [min]

### Derivadas (Engenharia):
- `temp_difference` - Diferença Térmica
- `estimated_power` - Potência Estimada
- `mechanical_stress` - Stress Mecânico
- `thermal_efficiency` - Eficiência Térmica
- `wear_per_operation` - Desgaste por Operação

## 🔧 Como Usar

```python
# Carregar features não normalizadas
df = pd.read_csv('data/processed/features_raw.csv')

# Separar features e target
X = df.drop('target', axis=1)
y = df['target']

# Normalizar APÓS train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 📈 Importância das Features

Com base no modelo Random Forest em produção:

1. **mechanical_stress** (19.5%) - Maior preditor
2. **rotational_speed** (16.7%) - Velocidade crítica  
3. **torque** (13.8%) - Força aplicada
4. **estimated_power** (12.9%) - Potência do sistema
5. **wear_per_operation** (12.1%) - Desgaste normalizado

## 🎯 Prevenção de Data Leakage

A arquitetura atual previne data leakage através de:

- ✅ Features extraídas sem normalização
- ✅ Normalização aplicada após train/test split
- ✅ Scaler treinado apenas nos dados de treino
- ✅ Mesma normalização aplicada na inferência 