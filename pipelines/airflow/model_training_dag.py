"""
DAG para Treinamento de Modelos de Manutenção Preditiva
Atualizado para trabalhar com train_model.py que já registra modelos no MLflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import logging

logger = logging.getLogger(__name__)

# Configurações padrão
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

# Definição do DAG
dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Pipeline de treinamento de modelos com MLflow 3.0',
    schedule_interval='@weekly',  # Executa semanalmente
    catchup=False,
    max_active_runs=1,
    tags=['machine-learning', 'predictive-maintenance', 'mlflow']
)

def validate_training_data(**context):
    """
    Valida se os dados estão prontos para treinamento
    """
    import pandas as pd
    import os
    from pathlib import Path
    
    logger.info("Validando dados de treinamento...")
    
    features_path = "/opt/airflow/data/processed/features_raw.csv"
    
    # Verifica se arquivo existe
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")
    
    # Carrega e valida dados
    try:
        df = pd.read_csv(features_path)
        
        # Validações básicas
        min_samples = 1000
        required_columns = ['target']
    
        validation = {
            'file_exists': True,
            'shape': df.shape,
            'min_samples_met': len(df) >= min_samples,
            'has_target': 'target' in df.columns,
            'missing_values': df.isnull().sum().sum(),
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {},
            'ready_for_training': True
        }
        
        # Verifica critérios
        if len(df) < min_samples:
            validation['ready_for_training'] = False
            logger.warning(f"Dataset muito pequeno: {len(df)} < {min_samples}")
            
        if 'target' not in df.columns:
            validation['ready_for_training'] = False
            logger.error("Coluna 'target' não encontrada")
            
        if validation['ready_for_training']:
            logger.info(f"✅ Dados validados: {df.shape[0]} amostras, {df.shape[1]} features")
            logger.info(f"   Distribuição do target: {validation['target_distribution']}")
        else:
            raise ValueError("Dados não atendem aos critérios mínimos para treinamento")
            
        # Salva informações para próximas tasks
        context['task_instance'].xcom_push(key='validation_results', value=validation)
    
        return validation

    except Exception as e:
        logger.error(f"Erro na validação dos dados: {e}")
        raise

def check_mlflow_results(**context):
    """
    Verifica os resultados do treinamento no MLflow e extrai informações dos modelos
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import time
    
    logger.info("Verificando resultados do treinamento no MLflow...")
    
    # Configura MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    try:
        # Busca o experimento mais recente
        experiments = client.search_experiments(order_by=["creation_time DESC"])
        
        if not experiments:
            raise ValueError("Nenhum experimento encontrado no MLflow")
        
        latest_experiment = experiments[0]
        experiment_id = latest_experiment.experiment_id
        
        logger.info(f"Analisando experimento: {latest_experiment.name} (ID: {experiment_id})")
        
        # Busca runs mais recentes (últimos 10 minutos)
        current_time = int(time.time() * 1000)
        ten_minutes_ago = current_time - (10 * 60 * 1000)
        
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.start_time >= {ten_minutes_ago}",
            order_by=["start_time DESC"],
            max_results=50
        )
        
        if not runs:
            raise ValueError("Nenhum run recente encontrado no experimento")
        
        logger.info(f"Encontrados {len(runs)} runs recentes")
        
        # Analisa runs para encontrar os principais
        training_results = {
            'experiment_id': experiment_id,
            'experiment_name': latest_experiment.name,
            'total_runs': len(runs),
            'main_run': None,
            'models_trained': [],
            'best_model': None
        }
        
        best_f1_score = 0
        main_run = None
        
        for run in runs:
            run_data = run.data
            run_info = run.info
            
            # Identifica run principal (parent run)
            if run_data.tags.get('experiment_type') == 'model_comparison':
                main_run = {
                    'run_id': run_info.run_id,
                    'run_name': run_data.tags.get('mlflow.runName', 'Unknown'),
                    'start_time': run_info.start_time,
                    'status': run_info.status,
                    'metrics': run_data.metrics,
                    'params': run_data.params
                }
                training_results['main_run'] = main_run
                
                # Extrai métricas do melhor modelo
                if 'best_model_f1_test' in run_data.metrics:
                    f1_score = run_data.metrics['best_model_f1_test']
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        training_results['best_model'] = {
                            'model_name': run_data.params.get('best_model', 'Unknown'),
                            'f1_score': f1_score,
                            'accuracy': run_data.metrics.get('best_model_accuracy_test', 0),
                            'train_test_diff': run_data.metrics.get('train_test_difference', 0),
                            'run_id': run_info.run_id
                        }
                
                logger.info(f"✅ Run principal encontrado: {main_run['run_name']}")
                break
        
        # Conta modelos treinados por tipo
        model_types = {}
        for run in runs:
            algorithm = run.data.tags.get('algorithm', 'unknown')
            if algorithm not in ['unknown']:
                if algorithm not in model_types:
                    model_types[algorithm] = 0
                model_types[algorithm] += 1
        
        training_results['models_trained'] = [
            {'algorithm': algo, 'count': count} 
            for algo, count in model_types.items()
        ]
        
        if training_results['best_model']:
            logger.info(f"🏆 Melhor modelo: {training_results['best_model']['model_name']}")
            logger.info(f"   F1-Score: {training_results['best_model']['f1_score']:.4f}")
            logger.info(f"   Accuracy: {training_results['best_model']['accuracy']:.4f}")
            logger.info(f"   Train-Test Diff: {training_results['best_model']['train_test_diff']:.4f}")
        else:
            logger.warning("Nenhum modelo com métricas encontrado")
        
        # Verifica se há modelos no Model Registry
        try:
            registered_models = client.search_registered_models()
            model_registry_info = []
            
            for model in registered_models:
                latest_versions = client.get_latest_versions(model.name)
                if latest_versions:
                    latest_version = latest_versions[0]
                    model_registry_info.append({
                        'name': model.name,
                        'latest_version': latest_version.version,
                        'stage': latest_version.current_stage,
                        'creation_timestamp': latest_version.creation_timestamp
                    })
            
            training_results['registered_models'] = model_registry_info
            logger.info(f"📋 Modelos registrados: {len(model_registry_info)}")
            
        except Exception as e:
            logger.warning(f"Erro ao verificar Model Registry: {e}")
            training_results['registered_models'] = []
        
        # Salva resultados para próximas tasks
        context['task_instance'].xcom_push(key='training_results', value=training_results)
        
        return training_results
        
    except Exception as e:
        logger.error(f"Erro ao verificar resultados do MLflow: {e}")
        raise

def evaluate_and_notify(**context):
    """
    Avalia os resultados e envia notificação consolidada
    """
    logger.info("Avaliando resultados e enviando notificação...")
    
    # Recupera dados das tasks anteriores
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    training_results = context['task_instance'].xcom_pull(key='training_results')
    
    if not training_results:
        raise ValueError("Resultados do treinamento não encontrados")
    
    # Critérios de qualidade
    min_f1_score = 0.85
    max_train_test_diff = 0.05
    
    evaluation = {
        'training_successful': True,
        'quality_passed': False,
        'production_ready': False,
        'recommendations': []
    }
    
    best_model = training_results.get('best_model')
    
    if best_model:
        f1_score = best_model['f1_score']
        train_test_diff = best_model['train_test_diff']
        
        # Avaliação de qualidade
        if f1_score >= min_f1_score:
            evaluation['quality_passed'] = True
            evaluation['recommendations'].append(f"✅ F1-Score excelente: {f1_score:.4f}")
        else:
            evaluation['recommendations'].append(f"⚠️ F1-Score abaixo do mínimo: {f1_score:.4f} < {min_f1_score}")
        
        if train_test_diff <= max_train_test_diff:
            evaluation['recommendations'].append(f"✅ Sem overfitting: diff {train_test_diff:.4f}")
        else:
            evaluation['recommendations'].append(f"⚠️ Possível overfitting: diff {train_test_diff:.4f} > {max_train_test_diff}")
        
        # Produção só se passou em ambos critérios
        evaluation['production_ready'] = (f1_score >= min_f1_score and train_test_diff <= max_train_test_diff)
    
    # Monta notificação
    message_parts = [
        "🤖 PIPELINE DE TREINAMENTO CONCLUÍDO",
        "=" * 50,
        f"📊 Experimento: {training_results.get('experiment_name', 'N/A')}",
        f"📈 Total de runs: {training_results.get('total_runs', 0)}",
        f"🔬 Modelos treinados: {', '.join([m['algorithm'] for m in training_results.get('models_trained', [])])}"
    ]
    
    if best_model:
        message_parts.extend([
            "",
            f"🏆 MELHOR MODELO: {best_model['model_name']}",
            f"   F1-Score: {best_model['f1_score']:.4f}",
            f"   Accuracy: {best_model['accuracy']:.4f}",
            f"   Train-Test Diff: {best_model['train_test_diff']:.4f}",
            ""
        ])
    
    message_parts.extend([
        "📋 AVALIAÇÃO:",
        f"   Qualidade: {'✅ APROVADO' if evaluation['quality_passed'] else '❌ REPROVADO'}",
        f"   Produção: {'✅ PRONTO' if evaluation['production_ready'] else '⚠️ NÃO PRONTO'}",
        ""
    ])
    
    if evaluation['recommendations']:
        message_parts.extend([
            "💡 RECOMENDAÇÕES:",
            *[f"   {rec}" for rec in evaluation['recommendations']],
            ""
        ])
    
    # Modelos registrados
    registered = training_results.get('registered_models', [])
    if registered:
        message_parts.extend([
            "📝 MODELOS REGISTRADOS:",
            *[f"   {m['name']} v{m['latest_version']} ({m['stage']})" for m in registered]
        ])
    
    final_message = "\n".join(message_parts)
    logger.info(final_message)
    
    # Salva avaliação final
    final_evaluation = {
        **evaluation,
        'message': final_message,
        'best_model': best_model,
        'timestamp': datetime.now().isoformat()
    }
    
    context['task_instance'].xcom_push(key='final_evaluation', value=final_evaluation)
    
    return final_evaluation

# ==================== DEFINIÇÃO DAS TASKS ====================

# Sensor para aguardar features prontas
features_sensor = FileSensor(
    task_id='wait_for_features',
    filepath='/opt/airflow/data/processed/features_raw.csv',
    fs_conn_id='fs_default',
    poke_interval=300,
    timeout=3600,
    dag=dag
)

# Task 1: Validação dos dados
validate_data_task = PythonOperator(
    task_id='validate_training_data',
    python_callable=validate_training_data,
    dag=dag
)

# Task 2: Treinamento do modelo (usando o script existente)
train_model_task = BashOperator(
    task_id='train_model',
    bash_command='cd /opt/airflow && python src/models/train_model.py',
    dag=dag
)

# Task 3: Verificação dos resultados no MLflow
check_results_task = PythonOperator(
    task_id='check_mlflow_results',
    python_callable=check_mlflow_results,
    dag=dag
)

# Task 4: Avaliação final e notificação
evaluate_notify_task = PythonOperator(
    task_id='evaluate_and_notify',
    python_callable=evaluate_and_notify,
    dag=dag
)

# ==================== DEFINIÇÃO DAS DEPENDÊNCIAS ====================
features_sensor >> validate_data_task >> train_model_task >> check_results_task >> evaluate_notify_task 