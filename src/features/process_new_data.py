"""
Pipeline de Processamento de Novos Dados para Manutenção Preditiva Industrial
Processa novos dados usando preprocessadores já treinados (sem data leakage)
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_preprocessing_objects():
    """
    Carrega objetos de preprocessamento treinados
    """
    try:
        scaler_path = '../../models/standard_scaler.pkl'
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Preprocessadores não encontrados. Execute o treinamento primeiro.")
        
        scaler = joblib.load(scaler_path)
        logger.info("Preprocessadores carregados com sucesso")
        
        return scaler
        
    except Exception as e:
        logger.error(f"Erro ao carregar preprocessadores: {e}")
        raise

def load_new_data():
    """
    Carrega novos dados para inferência
    """
    try:
        data_path = '../../data/inference/new_data.csv'
        
        if not os.path.exists(data_path):
            logger.warning(f"Arquivo de novos dados não encontrado: {data_path}")
            logger.info("Criando dados de exemplo...")
            create_sample_data()
        
        df = pd.read_csv(data_path)
        logger.info(f"Novos dados carregados: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar novos dados: {e}")
        raise

def create_sample_data():
    """
    Cria dados de exemplo para inferência baseados no dataset AI4I 2020
    """
    os.makedirs('../../data/inference', exist_ok=True)
    
    # Dados de exemplo baseados nas características do Type L
    sample_data = pd.DataFrame({
        'air_temperature': [298.1, 299.2, 300.5, 297.8, 301.2, 298.9, 299.7, 300.1, 298.5, 299.8],
        'process_temperature': [308.6, 309.8, 311.2, 307.9, 312.1, 309.5, 310.3, 310.8, 308.9, 310.5],
        'rotational_speed': [1551, 1476, 1682, 1398, 1728, 1523, 1605, 1567, 1489, 1634],
        'torque': [42.8, 38.9, 51.2, 35.7, 54.8, 41.3, 46.5, 44.1, 39.8, 48.3],
        'tool_wear': [108, 145, 67, 201, 89, 156, 92, 134, 178, 112]
    })
    
    sample_data.to_csv('../../data/inference/new_data.csv', index=False)
    logger.info("Dados de exemplo criados em: data/inference/new_data.csv")

def validate_input_data(df):
    """
    Valida dados de entrada para inferência
    """
    logger.info("Validando dados de entrada...")
    
    required_columns = [
        'air_temperature',
        'process_temperature', 
        'rotational_speed',
        'torque',
        'tool_wear'
    ]
    
    # Verifica se todas as colunas necessárias estão presentes
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas faltando: {missing_columns}")
    
    # Verifica valores ausentes
    missing_values = df[required_columns].isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Valores ausentes encontrados:\n{missing_values[missing_values > 0]}")
        # Preenche valores ausentes com mediana
        for col in required_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Preenchido {col} com mediana: {median_val}")
    
    # Verifica ranges válidos baseados no dataset original
    validation_ranges = {
        'air_temperature': (295, 305),  # Kelvin
        'process_temperature': (305, 315),  # Kelvin
        'rotational_speed': (1100, 3000),  # RPM
        'torque': (3, 80),  # Nm
        'tool_wear': (0, 300)  # minutes
    }
    
    for col, (min_val, max_val) in validation_ranges.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                logger.warning(f"{col}: {out_of_range} valores fora do range [{min_val}, {max_val}]")
    
    logger.info("Validação de dados de entrada concluída")
    return df

def extract_features(df):
    """
    Extrai features dos novos dados (mesmo processo usado no treinamento)
    """
    logger.info("Extraindo features dos novos dados...")
    
    features_df = pd.DataFrame()
    
    # Features principais (sensores)
    features_df['air_temperature'] = df['air_temperature']
    features_df['process_temperature'] = df['process_temperature']
    features_df['rotational_speed'] = df['rotational_speed']
    features_df['torque'] = df['torque']
    features_df['tool_wear'] = df['tool_wear']
    
    # Features derivadas (mesmas usadas no treinamento)
    # Diferença entre temperaturas (indica stress térmico)
    features_df['temp_difference'] = features_df['process_temperature'] - features_df['air_temperature']
    
    # Power estimado (Torque * Velocidade Angular)
    features_df['estimated_power'] = features_df['torque'] * (features_df['rotational_speed'] * 2 * np.pi / 60)
    
    # Stress mecânico (Torque / RPM ratio)
    features_df['mechanical_stress'] = features_df['torque'] / (features_df['rotational_speed'] / 1000)
    
    # Eficiência térmica estimada
    features_df['thermal_efficiency'] = features_df['process_temperature'] / features_df['air_temperature']
    
    # Desgaste normalizado por operação
    features_df['wear_per_operation'] = features_df['tool_wear'] / (features_df['rotational_speed'] * 0.001)
    
    logger.info(f"Features extraídas: {features_df.shape[1]} features")
    
    return features_df

def apply_preprocessing(df, scaler):
    """
    Aplica preprocessamento usando objetos treinados
    """
    logger.info("Aplicando preprocessamento...")
    
    try:
        # Aplica normalização usando scaler treinado
        features_scaled = scaler.transform(df)
        
        # Converte de volta para DataFrame
        df_scaled = pd.DataFrame(features_scaled, columns=df.columns, index=df.index)
        
        logger.info("Preprocessamento aplicado com sucesso")
        return df_scaled
        
    except Exception as e:
        logger.error(f"Erro ao aplicar preprocessamento: {e}")
        raise

def save_processed_data(df, filename='processed_data.csv'):
    """
    Salva dados processados para inferência
    """
    os.makedirs('../../data/inference', exist_ok=True)
    filepath = f'../../data/inference/{filename}'
    
    df.to_csv(filepath, index=False)
    logger.info(f"Dados processados salvos em: {filepath}")
    
    return filepath

def generate_summary_report(original_df, processed_df):
    """
    Gera relatório de resumo do processamento
    """
    logger.info("Gerando relatório de resumo...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'samples_processed': int(len(original_df)),
        'features_extracted': int(processed_df.shape[1]),
        'data_quality': {
            'missing_values': int(original_df.isnull().sum().sum()),
            'outliers_detected': 0  # Será implementado se necessário
        },
        'feature_statistics': {
            col: {
                'mean': float(processed_df[col].mean()),
                'std': float(processed_df[col].std()),
                'min': float(processed_df[col].min()),
                'max': float(processed_df[col].max())
            } for col in processed_df.columns
        }
    }
    
    # Salva relatório
    import json
    os.makedirs('../../data/inference', exist_ok=True)
    with open('../../data/inference/processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Relatório de resumo salvo em: data/inference/processing_report.json")
    
    return report

def main():
    """
    Pipeline principal de processamento de novos dados
    """
    logger.info("Iniciando processamento de novos dados para inferência...")
    
    try:
        # 1. Carrega objetos de preprocessamento treinados
        scaler = load_preprocessing_objects()
        
        # 2. Carrega novos dados
        df = load_new_data()
        
        # 3. Valida dados de entrada
        df = validate_input_data(df)
        
        # 4. Extrai features (mesmo processo do treinamento)
        features_df = extract_features(df)
        
        # 5. Aplica preprocessamento usando objetos treinados
        processed_df = apply_preprocessing(features_df, scaler)
        
        # 6. Salva dados processados
        save_processed_data(processed_df)
        
        # 7. Gera relatório de resumo
        report = generate_summary_report(df, processed_df)
        
        logger.info("Processamento de novos dados concluído com sucesso!")
        logger.info(f"Amostras processadas: {len(processed_df)}")
        logger.info(f"Features geradas: {processed_df.shape[1]}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Erro no processamento de novos dados: {e}")
        raise

if __name__ == "__main__":
    main()