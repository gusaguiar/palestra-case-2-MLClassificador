"""
Pipeline de Feature Engineering para Manutenção Preditiva Industrial
Processa dados históricos do dataset AI4I 2020 (Type L) para treinamento de modelos
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data():
    """
    Carrega dados brutos filtrados (Type L) do dataset AI4I 2020
    """
    try:
        data_path = '/opt/airflow/data/raw/ai4i2020_type_l.csv'
        if not os.path.exists(data_path):
            logger.info("Arquivo Type L não encontrado. Filtrando dataset original...")
            
            # Carrega dataset original e filtra Type L
            original_path = '/opt/airflow/data/raw/ai4i2020.csv'
            df_original = pd.read_csv(original_path)
            df = df_original[df_original['Type'] == 'L'].copy()
            
            # Salva dataset filtrado
            df.to_csv(data_path, index=False)
            logger.info(f"Dataset Type L salvo: {df.shape[0]} amostras")
        else:
            df = pd.read_csv(data_path)
            
        logger.info(f"Dados carregados: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def validate_data_quality(df):
    """
    Valida qualidade dos dados de entrada
    """
    logger.info("Validando qualidade dos dados...")
    
    # Verifica valores ausentes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Valores ausentes encontrados:\n{missing_values[missing_values > 0]}")
    
    # Verifica duplicatas
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Linhas duplicadas encontradas: {duplicates}")
        df = df.drop_duplicates()
    
    # Estatísticas básicas das features principais
    feature_columns = [
        'Air temperature [K]',
        'Process temperature [K]', 
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    logger.info("Estatísticas das features:")
    for col in feature_columns:
        if col in df.columns:
            logger.info(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    # Valida distribuição do target
    target_dist = df['Machine failure'].value_counts()
    logger.info(f"Distribuição do target:\n{target_dist}")
    logger.info(f"Taxa de falhas: {df['Machine failure'].mean():.4f}")
    
    return df

def extract_features(df):
    """
    Extrai e processa features para manutenção preditiva
    """
    logger.info("Extraindo features...")
    
    features_df = pd.DataFrame()
    
    # Features principais (sensores)
    features_df['air_temperature'] = df['Air temperature [K]']
    features_df['process_temperature'] = df['Process temperature [K]']
    features_df['rotational_speed'] = df['Rotational speed [rpm]']
    features_df['torque'] = df['Torque [Nm]']
    features_df['tool_wear'] = df['Tool wear [min]']
    
    # Features derivadas
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
    
    # Target
    features_df['target'] = df['Machine failure']
    
    logger.info(f"Features extraídas: {features_df.shape[1]-1} features + target")
    
    return features_df

def detect_outliers(df, features):
    """
    Detecta outliers usando IQR
    """
    logger.info("Detectando outliers...")
    
    outlier_counts = {}
    
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound))
        outlier_counts[feature] = outliers.sum()
        
        if outliers.sum() > 0:
            logger.info(f"{feature}: {outliers.sum()} outliers ({outliers.sum()/len(df)*100:.2f}%)")
    
    return outlier_counts

def validate_features(df):
    """
    Valida features extraídas
    """
    logger.info("Validando features extraídas...")
    
    feature_columns = df.columns.drop('target')
    
    # Verifica correlações altas
    corr_matrix = df[feature_columns].corr()
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        logger.warning("Correlações altas detectadas:")
        for feat1, feat2, corr in high_corr:
            logger.warning(f"{feat1} - {feat2}: {corr:.3f}")
    
    # Detecta outliers
    outlier_counts = detect_outliers(df, feature_columns)
    
    # Verifica variância
    low_variance = []
    for col in feature_columns:
        if df[col].var() < 0.01:
            low_variance.append(col)
    
    if low_variance:
        logger.warning(f"Features com baixa variância: {low_variance}")
    
    logger.info("Validação de features concluída")
    
    return df

def save_features(df, filename):
    """
    Salva features processadas sem normalização (para evitar data leakage)
    """
    os.makedirs('/opt/airflow/data/processed', exist_ok=True)
    filepath = f'/opt/airflow/data/processed/{filename}'
    
    df.to_csv(filepath, index=False)
    logger.info(f"Features salvas em: {filepath}")
    
    return filepath

def save_preprocessing_objects():
    """
    Salva objetos de preprocessamento (serão treinados durante o treinamento)
    """
    os.makedirs('/opt/airflow/models', exist_ok=True)
    
    # Cria scalers vazios que serão treinados posteriormente
    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()
    
    # Salva scalers não treinados
    joblib.dump(scaler_standard, '/opt/airflow/models/standard_scaler.pkl')
    joblib.dump(scaler_robust, '/opt/airflow/models/robust_scaler.pkl')
    
    logger.info("Objetos de preprocessamento criados e salvos")

def main():
    """
    Pipeline principal de feature engineering
    """
    logger.info("Iniciando pipeline de feature engineering...")
    
    try:
        # 1. Carrega dados brutos
        df = load_raw_data()
        
        # 2. Valida qualidade dos dados
        df = validate_data_quality(df)
        
        # 3. Extrai features
        features_df = extract_features(df)
        
        # 4. Valida features
        features_df = validate_features(features_df)
        
        # 5. Salva features (sem normalização para evitar data leakage)
        save_features(features_df, 'features_raw.csv')
        
        # 6. Salva objetos de preprocessamento
        save_preprocessing_objects()
        
        logger.info("Pipeline de feature engineering concluída com sucesso!")
        logger.info(f"Dataset final: {features_df.shape[0]} amostras, {features_df.shape[1]-1} features")
        
        # Estatísticas finais
        logger.info(f"Taxa de falhas no dataset: {features_df['target'].mean():.4f}")
        
    except Exception as e:
        logger.error(f"Erro na pipeline de feature engineering: {e}")
        raise

if __name__ == "__main__":
    main() 