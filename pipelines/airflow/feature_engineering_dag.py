"""
DAG do Airflow para Feature Engineering - Manutenção Preditiva Industrial
Processa dados do dataset AI4I 2020 Type L para predição de falhas em equipamentos
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações padrão do DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['ml-team@company.com']
}

# Definição do DAG
dag = DAG(
    'feature_engineering_pipeline',
    default_args=default_args,
    description='Pipeline de Feature Engineering para Manutenção Preditiva Industrial (AI4I 2020 Type L)',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'feature-engineering', 'maintenance-prediction', 'industrial']
)

def check_raw_data_quality(**context):
    """
    Verifica a qualidade dos dados brutos do dataset AI4I 2020 Type L
    """
    import pandas as pd
    from pathlib import Path
    
    logger.info("Verificando qualidade dos dados AI4I 2020 Type L...")
    
    # Verifica se o arquivo de dados existe
    data_path = Path("/opt/airflow/data/raw/ai4i2020_type_l.csv")
    if not data_path.exists():
        # Tenta carregar dataset original e filtrar
        original_path = Path("/opt/airflow/data/raw/ai4i2020.csv")
        if not original_path.exists():
            raise FileNotFoundError(f"Dataset AI4I 2020 não encontrado: {original_path}")
        
        logger.info("Filtrando dataset original para Type L...")
        df_original = pd.read_csv(original_path)
        df = df_original[df_original['Type'] == 'L'].copy()
        df.to_csv(data_path, index=False)
        logger.info(f"Dataset Type L criado: {len(df)} amostras")
    else:
        df = pd.read_csv(data_path)
    
    # Verificações de qualidade específicas para manutenção preditiva
    required_columns = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Machine failure'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias faltando: {missing_columns}")
    
    checks = {
        'total_samples': len(df),
        'failure_rate': df['Machine failure'].mean(),
        'missing_values': df[required_columns].isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'feature_ranges': {
            'air_temp_range': [df['Air temperature [K]'].min(), df['Air temperature [K]'].max()],
            'process_temp_range': [df['Process temperature [K]'].min(), df['Process temperature [K]'].max()],
            'rpm_range': [df['Rotational speed [rpm]'].min(), df['Rotational speed [rpm]'].max()],
            'torque_range': [df['Torque [Nm]'].min(), df['Torque [Nm]'].max()],
            'wear_range': [df['Tool wear [min]'].min(), df['Tool wear [min]'].max()]
        }
    }
    
    logger.info(f"Verificações de qualidade: {checks}")
    
    # Critérios de qualidade para manutenção preditiva
    if checks['total_samples'] < 5000:
        raise ValueError(f"Dados insuficientes: {checks['total_samples']} amostras (mínimo 5000)")
    
    if checks['failure_rate'] < 0.01 or checks['failure_rate'] > 0.1:
        logger.warning(f"Taxa de falhas incomum: {checks['failure_rate']:.4f}")
    
    if checks['missing_values'] > 0:
        logger.warning(f"Valores ausentes encontrados: {checks['missing_values']}")
    
    # Salva relatório de qualidade
    context['task_instance'].xcom_push(key='data_quality_report', value=checks)
    
    logger.info("Dados AI4I 2020 Type L passaram na verificação de qualidade!")
    return checks

def validate_extracted_features(**context):
    """
    Valida as features extraídas para manutenção preditiva (RAW - sem normalização para evitar data leakage)
    """
    import pandas as pd
    from pathlib import Path
    import numpy as np
    
    logger.info("Validando features RAW extraídas para manutenção preditiva...")
    
    # Verifica arquivos de features RAW (não normalizadas)
    features_path = Path("/opt/airflow/data/processed/features_raw.csv")
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features RAW não encontradas: {features_path}")
    
    # Carrega e valida
    features = pd.read_csv(features_path)
    
    # Extrai target das features raw
    if 'target' not in features.columns:
        raise ValueError("Coluna 'target' não encontrada nas features")
    
    target = features['target']
    
    # Separa features numéricas do target
    feature_columns = [col for col in features.columns if col != 'target']
    
    numeric_features = features[feature_columns]
    
    # Verifica features esperadas para manutenção preditiva
    expected_features = [
        'air_temperature',
        'process_temperature',
        'rotational_speed',
        'torque',
        'tool_wear'
    ]
    
    missing_features = set(expected_features) - set(feature_columns)
    if missing_features:
        logger.warning(f"Features esperadas faltando: {missing_features}")
    
    validation = {
        'features_shape': features.shape,
        'numeric_features_count': len(feature_columns),
        'features_with_nan': numeric_features.isnull().sum().sum(),
        'features_with_inf': np.isinf(numeric_features).sum().sum(),
        'total_samples': features.shape[0],
        'failure_rate': target.mean(),
        'target_distribution': target.value_counts().to_dict(),
        'unique_classes': target.nunique(),
        'data_leakage_safe': True,  # Features RAW não normalizadas
        'expected_features_present': len(missing_features) == 0,
        'feature_statistics': {
            col: {
                'mean': float(numeric_features[col].mean()),
                'std': float(numeric_features[col].std()),
                'min': float(numeric_features[col].min()),
                'max': float(numeric_features[col].max())
            } for col in feature_columns[:5]  # Top 5 features
        }
    }
    
    logger.info(f"Validação de features RAW: {validation}")
    
    # Critérios de validação para manutenção preditiva
    if validation['features_with_nan'] > 0:
        logger.warning(f"Features contêm valores NaN: {validation['features_with_nan']} (serão preenchidos)")
    
    if validation['features_with_inf'] > 0:
        logger.warning(f"Features contêm valores infinitos: {validation['features_with_inf']} (serão substituídos)")
    
    if validation['numeric_features_count'] < 5:
        raise ValueError(f"Poucas features numéricas extraídas: {validation['numeric_features_count']} (mínimo 5)")
    
    if validation['unique_classes'] != 2:
        raise ValueError(f"Problema de classificação binária deve ter 2 classes, encontradas: {validation['unique_classes']}")
    
    if validation['failure_rate'] < 0.01 or validation['failure_rate'] > 0.2:
        logger.warning(f"Taxa de falhas incomum: {validation['failure_rate']:.4f}")
    
    # Salva relatório de validação
    context['task_instance'].xcom_push(key='feature_validation_report', value=validation)
    
    logger.info("Features RAW para manutenção preditiva validadas com sucesso (sem data leakage)!")
    return validation

def clean_old_features(**context):
    """
    Limpa features antigas e arquivos temporários para economizar espaço
    """
    from pathlib import Path
    import shutil
    from datetime import datetime, timedelta
    
    logger.info("Limpando features antigas e arquivos temporários...")
    
    # Diretórios para limpeza
    dirs_to_clean = [
        Path("data/processed"),
        Path("data/inference"),
        Path("models")
    ]
    
    # Remove arquivos temporários com mais de 7 dias
    cutoff_date = datetime.now() - timedelta(days=7)
    files_removed = 0
    
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
            
        # Remove arquivos temporários
        for pattern in ["*_temp_*", "*_backup_*", "*.tmp"]:
            for file_path in dir_path.glob(pattern):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        files_removed += 1
                        logger.info(f"Removido arquivo antigo: {file_path}")
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo {file_path}: {e}")
    
    # Remove logs antigos de processamento
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    files_removed += 1
                    logger.info(f"Removido log antigo: {log_file}")
            except Exception as e:
                logger.warning(f"Erro ao remover log {log_file}: {e}")
    
    logger.info(f"Limpeza concluída! {files_removed} arquivos removidos.")

# Sensor para verificar se há dados AI4I 2020
data_sensor = FileSensor(
    task_id='wait_for_ai4i_data',
    filepath='/opt/airflow/data/raw/ai4i2020.csv',
    fs_conn_id='fs_default',
    poke_interval=300,  # Verifica a cada 5 minutos
    timeout=3600,  # Timeout de 1 hora
    dag=dag
)

# Task 1: Verificação de qualidade dos dados brutos
data_quality_task = PythonOperator(
    task_id='check_raw_data_quality',
    python_callable=check_raw_data_quality,
    dag=dag
)

# Task 2: Carregamento de dados
load_data_task = BashOperator(
    task_id='load_data',
    bash_command='cd /opt/airflow && python src/data/load_data.py',
    dag=dag
)

# Task 3: Feature Engineering
feature_engineering_task = BashOperator(
    task_id='extract_features',
    bash_command='cd /opt/airflow && python src/features/build_features.py',
    dag=dag
)

# Task 4: Validação de features
validate_features_task = PythonOperator(
    task_id='validate_extracted_features',
    python_callable=validate_extracted_features,
    dag=dag
)

# Task 5: Limpeza de arquivos antigos
cleanup_task = PythonOperator(
    task_id='clean_old_features',
    python_callable=clean_old_features,
    dag=dag
)

# Definição das dependências
data_sensor >> data_quality_task >> load_data_task
load_data_task >> feature_engineering_task >> validate_features_task
validate_features_task >> cleanup_task 