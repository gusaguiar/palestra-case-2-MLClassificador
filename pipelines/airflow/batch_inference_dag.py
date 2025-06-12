"""
DAG do Airflow para Inferência em Lote (Batch Inference)
Atualizado para carregar scaler do MLflow e usar o nome correto do modelo
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
    'batch_inference_pipeline',
    default_args=default_args,
    description='Pipeline de Inferência em Lote com MLflow Model Registry e Scaler',
    schedule_interval='@daily',  # Executa diariamente
    catchup=False,
    tags=['ml', 'inference', 'batch', 'mlflow']
)

def validate_inference_data(**context):
    """
    Valida os dados para inferência
    """
    import pandas as pd
    from pathlib import Path
    
    logger.info("Validando dados para inferência...")
    
    # Verifica se há dados novos
    inference_data_path = Path("/opt/airflow/data/inference/new_data.csv")
    
    if not inference_data_path.exists():
        raise FileNotFoundError(f"Dados para inferência não encontrados: {inference_data_path}")
    
    # Carrega e valida
    data = pd.read_csv(inference_data_path)
    
    validation = {
        'total_samples': len(data),
        'shape': data.shape,
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'file_size_mb': inference_data_path.stat().st_size / (1024 * 1024)
    }
    
    logger.info(f"Dados validados: {validation}")
    
    # Critérios de validação
    if validation['total_samples'] == 0:
        raise ValueError("Arquivo de dados está vazio")
    
    if validation['missing_values'] > validation['total_samples'] * 0.5:
        raise ValueError(f"Muitos valores ausentes: {validation['missing_values']}")
    
    context['task_instance'].xcom_push(key='data_validation', value=validation)
    
    return validation

def load_production_model_and_preprocessor(**context):
    """
    Carrega modelo em produção E scaler do MLflow Model Registry
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.pyfunc import load_model
    import joblib
    import tempfile
    import os
    
    logger.info("Carregando modelo e preprocessador em produção...")

    # Configurar o MLflow tracking URI
    # Tenta primeiro o Model Registry local onde os modelos estão registrados  
    local_mlruns_path = '/opt/airflow/src/models/mlruns'
    if os.path.exists(os.path.join(local_mlruns_path, 'models', 'predictive_maintenance_model')):
        mlflow_tracking_uri = f'file://{local_mlruns_path}'
        logger.info(f"Usando Model Registry local: {mlflow_tracking_uri}")
    else:
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        logger.info(f"Usando MLflow server: {mlflow_tracking_uri}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    # Nome correto do modelo conforme definido no train_model.py
    model_name = "predictive_maintenance_model"

    try:
        # Método alternativo: Carregar modelo diretamente dos arquivos físicos
        # para evitar problemas de caminho Windows/Linux
        
        # Buscar modelo físico disponível
        model_found = False
        model = None
        scaler = None
        
        # Procurar pelo Random Forest (melhor modelo baseado na memória)
        rf_paths = [
            '/opt/airflow/src/models/mlruns/906439392966595488/c2044e9d268a4844a7cd4e245f4db12e/artifacts/random_forest/model.pkl',
            '/opt/airflow/src/models/mlruns/960816347499315434/ef9f80f053d7455cbb7aaa460ce3a6ba/artifacts/random_forest/model.pkl'
        ]
        
        for model_path in rf_paths:
            if os.path.exists(model_path):
                logger.info(f"Carregando Random Forest de: {model_path}")
                model = joblib.load(model_path)
                model_found = True
                model_type = "random_forest"
                break
        
        if not model_found:
            # Fallback: procurar SVM
            svm_paths = [
                '/opt/airflow/src/models/mlruns/906439392966595488/6c0e67274dde413bb886394c98507b3a/artifacts/svm/model.pkl',
                '/opt/airflow/src/models/mlruns/960816347499315434/88811a5071a84ae18f2c90e3a150ef4a/artifacts/svm/model.pkl'
            ]
            
            for model_path in svm_paths:
                if os.path.exists(model_path):
                    logger.info(f"Carregando SVM de: {model_path}")
                    model = joblib.load(model_path)
                    model_found = True
                    model_type = "svm"
                    break
        
        if not model_found:
            raise ValueError("Nenhum modelo encontrado nos caminhos esperados")
        
        # Carregar scaler local (já disponível)
        local_scaler_path = '/opt/airflow/models/scaler.pkl'
        if os.path.exists(local_scaler_path):
            scaler = joblib.load(local_scaler_path)
            logger.info("Scaler local carregado com sucesso")
        else:
            raise ValueError("Scaler não encontrado")
        
        model_info = {
            'model_name': f"{model_type}_predictive_maintenance",
            'version': 'local_file',
            'stage': 'Production',
            'model_uri': model_path,
            'loaded_successfully': True,
            'scaler_loaded': scaler is not None,
            'tracking_uri': mlflow_tracking_uri,
            'load_method': 'direct_file'
        }
        
        logger.info(f"Modelo e preprocessador carregados: {model_info}")
        
        context['task_instance'].xcom_push(key='model_info', value=model_info)
        
        # Salva modelo e scaler temporariamente para uso nas próximas tarefas
        joblib.dump(model, "temp_model.pkl")
        if scaler is not None:
            joblib.dump(scaler, "temp_scaler.pkl")
        
        return model_info
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

def prepare_features_for_inference(**context):
    """
    Prepara features para inferência usando o mesmo pipeline de features E aplica scaler
    """
    import pandas as pd
    import sys
    import os
    import joblib
    
    # Adiciona o diretório src ao path para importar módulos
    sys.path.append('/opt/airflow/src')
    
    logger.info("Preparando features para inferência...")
    
    # Carrega dados novos
    input_file = "/opt/airflow/data/inference/new_data.csv"
    output_file = "/opt/airflow/data/inference/features_for_inference.csv"
    
    try:
        data = pd.read_csv(input_file)
        
        # Verifica se os dados têm as colunas esperadas do dataset AI4I
        expected_columns = [
            'Air temperature [K]',
            'Process temperature [K]', 
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Colunas ausentes nos dados de entrada: {missing_columns}")
            logger.info(f"Colunas disponíveis: {list(data.columns)}")
            
            # Mapeamento fixo e correto
            column_mapping = {
                'air_temperature': 'Air temperature [K]',
                'process_temperature': 'Process temperature [K]', 
                'rotational_speed': 'Rotational speed [rpm]',
                'torque': 'Torque [Nm]',
                'tool_wear': 'Tool wear [min]'
            }
            
            logger.info(f"Aplicando mapeamento de colunas: {column_mapping}")
            data = data.rename(columns=column_mapping)
        
        # Aplica o mesmo pipeline de feature engineering do treinamento
        from features.build_features import extract_features
        
        # Adiciona coluna de target dummy (será removida depois)
        data['Machine failure'] = 0
        
        # Extrai features usando a mesma função do pipeline de treinamento
        features_df = extract_features(data)
        
        # Remove o target (dummy)
        if 'target' in features_df.columns:
            features_df = features_df.drop('target', axis=1)
        
        # IMPORTANTE: Aplica o scaler antes de salvar
        if os.path.exists("temp_scaler.pkl"):
            scaler = joblib.load("temp_scaler.pkl")
            
            # Aplica a normalização nas features numéricas
            numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns
            features_df[numeric_features] = scaler.transform(features_df[numeric_features])
            
            logger.info("Features normalizadas usando o scaler do modelo")
        else:
            logger.warning("Scaler não encontrado - dados NÃO foram normalizados!")
        
        # Salva features preparadas e normalizadas
        features_df.to_csv(output_file, index=False)
        
        preparation_info = {
            'input_samples': len(data),
            'original_columns': list(data.columns),
            'features_count': len(features_df.columns),
            'samples_count': len(features_df),
            'output_file': output_file,
            'scaler_applied': os.path.exists("temp_scaler.pkl"),
            'feature_engineering_applied': True
        }
        
        logger.info(f"Features preparadas: {preparation_info}")
        
        context['task_instance'].xcom_push(key='feature_preparation', value=preparation_info)
        
        return preparation_info
        
    except Exception as e:
        logger.error(f"Erro ao preparar features: {e}")
        raise

def run_batch_inference(**context):
    """
    Executa inferência em lote usando o modelo de produção com dados PRÉ-PROCESSADOS
    """
    import pandas as pd
    import joblib
    import numpy as np
    from datetime import datetime
    
    logger.info("Executando inferência em lote...")
    
    # Carrega modelo de produção (salvo pela task anterior)
    model = joblib.load('temp_model.pkl')
    
    # Carrega features processadas E NORMALIZADAS
    features_df = pd.read_csv("/opt/airflow/data/inference/features_for_inference.csv")
    
    # IMPORTANTE: Os dados já estão normalizados pela task anterior
    X = features_df.select_dtypes(include=[np.number])
    
    try:
        # Faz predições (dados já estão normalizados)
        predictions = model.predict(X)
        prediction_probabilities = model.predict_proba(X)
        
        # Cria DataFrame com resultados
        results_df = features_df.copy()
        results_df['predicted_class'] = predictions
        
        # Adiciona probabilidades para cada classe
        classes = model.classes_ if hasattr(model, 'classes_') else range(prediction_probabilities.shape[1])
        for i, class_label in enumerate(classes):
            results_df[f'prob_class_{class_label}'] = prediction_probabilities[:, i]
        
        # Adiciona confiança (maior probabilidade)
        results_df['confidence'] = np.max(prediction_probabilities, axis=1)
        
        # Adiciona metadados
        results_df['inference_timestamp'] = datetime.now().isoformat()
        model_info = context['task_instance'].xcom_pull(key='model_info')
        results_df['model_version'] = model_info['version']
        results_df['model_stage'] = model_info['stage']
        
        # Salva resultados
        output_path = f"/opt/airflow/data/inference/batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        
        # Estatísticas das predições
        prediction_stats = {
            'total_predictions': len(results_df),
            'unique_classes': len(np.unique(predictions)),
            'class_distribution': pd.Series(predictions).value_counts().to_dict(),
            'average_confidence': float(np.mean(results_df['confidence'])),
            'low_confidence_count': int(np.sum(results_df['confidence'] < 0.7)),
            'output_file': output_path,
            'preprocessing_applied': context['task_instance'].xcom_pull(key='feature_preparation')['scaler_applied']
        }
        
        logger.info(f"Inferência concluída: {prediction_stats}")
        
        context['task_instance'].xcom_push(key='inference_results', value=prediction_stats)
        
        return prediction_stats
        
    except Exception as e:
        logger.error(f"Erro durante inferência: {e}")
        raise

def generate_inference_report(**context):
    """
    Gera relatório de inferência
    """
    import pandas as pd
    
    logger.info("Gerando relatório de inferência...")
    
    # Recupera informações das tarefas anteriores
    model_info = context['task_instance'].xcom_pull(key='model_info')
    data_validation = context['task_instance'].xcom_pull(key='data_validation')
    feature_preparation = context['task_instance'].xcom_pull(key='feature_preparation')
    inference_results = context['task_instance'].xcom_pull(key='inference_results')
    
    # Carrega resultados detalhados
    results_file = inference_results['output_file']
    results_df = pd.read_csv(results_file)
    
    # Cria relatório
    report = f"""
    🔮 RELATÓRIO DE INFERÊNCIA EM LOTE
    
    Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    🤖 MODELO UTILIZADO:
    - Nome: {model_info['model_name']}
    - Versão: {model_info['version']}
    - Stage: {model_info['stage']}
    
    📊 DADOS PROCESSADOS:
    - Amostras de entrada: {data_validation['total_samples']}
    - Features extraídas: {feature_preparation['features_count']}
    - Amostras processadas: {feature_preparation['samples_count']}
    
    🎯 RESULTADOS DA INFERÊNCIA:
    - Total de predições: {inference_results['total_predictions']}
    - Classes únicas: {inference_results['unique_classes']}
    - Confiança média: {inference_results['average_confidence']:.3f}
    - Predições de baixa confiança: {inference_results['low_confidence_count']}
    
    📈 DISTRIBUIÇÃO DAS CLASSES:
    """
    
    for class_label, count in inference_results['class_distribution'].items():
        percentage = (count / inference_results['total_predictions']) * 100
        report += f"    - Classe {class_label}: {count} ({percentage:.1f}%)\n"
    
    report += f"""
    📁 ARQUIVO DE SAÍDA:
    - {results_file}
    
    ⚠️  ALERTAS:
    """
    
    # Adiciona alertas se necessário
    if inference_results['low_confidence_count'] > inference_results['total_predictions'] * 0.1:
        report += f"    - Alto número de predições com baixa confiança ({inference_results['low_confidence_count']})\n"
    
    if inference_results['average_confidence'] < 0.8:
        report += f"    - Confiança média baixa ({inference_results['average_confidence']:.3f})\n"
    
    if len(inference_results['class_distribution']) == 1:
        report += "    - Todas as predições pertencem à mesma classe\n"
    
    logger.info(report)
    
    # Salva relatório
    report_path = f"/opt/airflow/data/inference/batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return {'report_path': report_path, 'report_content': report}

def cleanup_temp_files(**context):
    """
    Limpa arquivos temporários
    """
    import os
    
    logger.info("Limpando arquivos temporários...")
    
    temp_files = [
        "temp_model.pkl",
        "temp_scaler.pkl",
        "/opt/airflow/data/inference/features_for_inference.csv"
    ]
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Removido: {temp_file}")
        except Exception as e:
            logger.warning(f"Erro ao remover {temp_file}: {e}")
    
    logger.info("Limpeza concluída!")

# Sensor para aguardar novos dados
new_data_sensor = FileSensor(
    task_id='wait_for_new_data',
    filepath='/opt/airflow/data/inference/new_data.csv',
    fs_conn_id='fs_default',
    poke_interval=300,
    timeout=3600,
    dag=dag
)

# Task 1: Validação dos dados
validate_data_task = PythonOperator(
    task_id='validate_inference_data',
    python_callable=validate_inference_data,
    dag=dag
)

# Task 2: Carregamento do modelo
load_model_task = PythonOperator(
    task_id='load_production_model_and_preprocessor',
    python_callable=load_production_model_and_preprocessor,
    dag=dag
)

# Task 3: Preparação de features
prepare_features_task = PythonOperator(
    task_id='prepare_features_for_inference',
    python_callable=prepare_features_for_inference,
    dag=dag
)

# Task 4: Inferência em lote
inference_task = PythonOperator(
    task_id='run_batch_inference',
    python_callable=run_batch_inference,
    dag=dag
)

# Task 5: Geração de relatório
report_task = PythonOperator(
    task_id='generate_inference_report',
    python_callable=generate_inference_report,
    dag=dag
)

# Task 6: Limpeza
cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag
)

# Definição das dependências
new_data_sensor >> validate_data_task
[validate_data_task, load_model_task] >> prepare_features_task
prepare_features_task >> inference_task >> report_task >> cleanup_task 