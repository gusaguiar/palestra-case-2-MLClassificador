"""
Treinamento de Modelos para Manutenção Preditiva Industrial
Classificação binária de falhas em equipamentos (Machine Failure)
Dataset: AI4I 2020 Type L com 6000 amostras
MLflow 3.0 - Melhores práticas com autologging e model registry moderno
"""

import pandas as pd
import numpy as np
import os
import time
import tempfile
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import mlflow
import mlflow.sklearn
import logging
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature, MetricThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do MLflow 3.0
mlflow.set_tracking_uri("http://mlflow:5000")

# Criar ou recuperar experimento com lógica robusta para experimentos deletados
def create_or_get_experiment():
    """
    Cria ou obtém experimento MLflow com fallback robusto para experimentos deletados
    """
    # Lista de nomes para tentar, em ordem de preferência
    experiment_candidates = [
        "predictive_maintenance",
        f"predictive_maintenance_{datetime.now().strftime('%Y%m%d')}",
        f"predictive_maintenance_{datetime.now().strftime('%Y%m%d_%H%M')}",
        f"predictive_maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        f"pump_fault_detection_{int(time.time())}"  # timestamp único como último recurso
    ]
    
    for experiment_name in experiment_candidates:
        try:
            logger.info(f"Tentando usar experimento: {experiment_name}")
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ Experimento ativo: {experiment_name}")
            return experiment_name
        except Exception as e:
            logger.warning(f"Erro ao acessar experimento '{experiment_name}': {e}")
            continue
    
    # Se chegou até aqui, algo está muito errado
    raise Exception("Não foi possível criar nenhum experimento MLflow")

# Usar a função robusta
active_experiment = create_or_get_experiment()

# Habilita autologging para scikit-learn (MLflow 3.0)
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5
)

def load_features():
    """
    Carrega features preprocessadas
    """
    logger.info("Carregando features...")
    
    features_path = "/opt/airflow/data/processed/features_raw.csv"
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features não encontradas em: {features_path}")
    
    df = pd.read_csv(features_path)
    logger.info(f"Features carregadas: {df.shape}")
    
    return df

def calculate_class_weights(y):
    """
    Calcula pesos proporcionais ao desbalanceamento das classes
    Para penalizar mais os erros nas classes menores
    """
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    # Cria dicionário de pesos
    weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    
    logger.info(f"Pesos calculados para penalização proporcional:")
    logger.info(f"  Classe 0 (Normal): peso {weight_dict[0]:.3f}")
    logger.info(f"  Classe 1 (Falha): peso {weight_dict[1]:.3f}")
    logger.info(f"  Penalização da classe minoritária: {weight_dict[1]/weight_dict[0]:.2f}x")
    
    return weight_dict

def create_feature_importance_plot(model, feature_names, model_name):
    """
    Cria gráfico de importância das features
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Modelo {model_name} não possui importâncias calculáveis")
        return None, None
    
    # Cria DataFrame para facilitar ordenação
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Cria o gráfico
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title(f'Importância das Features - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importância', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Adiciona valores nas barras
    for bar, value in zip(bars, importance_df['importance']):
        plt.text(value, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Salva o gráfico em diretório temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plot_path = tmp_file.name
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de importância salvo: {plot_path}")
    
    # Log das importâncias
    logger.info(f"Top 5 features mais importantes para {model_name}:")
    for idx, row in importance_df.tail(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f} ({row['importance']/importance_df['importance'].sum()*100:.1f}%)")
    
    return plot_path, importance_df

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """
    Cria gráfico da matriz de confusão
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Falha'],
                yticklabels=['Normal', 'Falha'],
                cbar_kws={'label': 'Contagem'})
    plt.title(f'Matriz de Confusão - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    
    # Adiciona métricas na matriz
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    plt.figtext(0.02, 0.02, f'Acurácia: {accuracy:.3f} | Precisão: {precision:.3f} | Recall: {recall:.3f}', 
                fontsize=10, ha='left')
    
    plt.tight_layout()
    
    # Salva o gráfico em diretório temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plot_path = tmp_file.name
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Matriz de confusão salva: {plot_path}")
    return plot_path

def create_roc_curve_plot(y_true, y_proba, model_name):
    """
    Cria gráfico da curva ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random (AUC = 0.5)')
    plt.fill_between(fpr, tpr, alpha=0.3)
    
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Adiciona grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva o gráfico em diretório temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plot_path = tmp_file.name
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Curva ROC salva: {plot_path}")
    return plot_path

def get_current_production_model():
    """
    Obtém o modelo atualmente em produção usando aliases do MLflow 3.0
    """
    client = MlflowClient()
    
    try:
        # Busca modelo usando alias 'champion' (substitui 'Production' stage)
        model_version = client.get_model_version_by_alias(
            name="predictive_maintenance_model", 
            alias="champion"
        )
        
        # Carrega modelo e obtém métricas
        model_uri = f"models:/predictive_maintenance_model@champion"
        
        # Busca métricas do run
        run = client.get_run(model_version.run_id)
        production_f1 = run.data.metrics.get('test_f1_score', 0.0)
        
        logger.info(f"Modelo em produção encontrado: v{model_version.version} (F1: {production_f1:.4f})")
        return production_f1, model_version
        
    except Exception as e:
        logger.info(f"Nenhum modelo em produção encontrado: {e}")
        return 0.0, None

def register_model_with_intelligent_promotion(model, model_name, test_f1_score, run_id, artifact_path="model"):
    """
    Registro inteligente no Model Registry com promoção baseada em performance
    Usa aliases do MLflow 3.0 ao invés de stages deprecated
    """
    client = MlflowClient()
    model_registry_name = "predictive_maintenance_model"
    
    try:
        # Obtém performance do modelo atual em produção
        production_f1, current_champion = get_current_production_model()
        
        # Define artifact_path baseado no nome do modelo se não especificado
        if artifact_path == "model":
            if "Random" in model_name or "RF" in model_name:
                artifact_path = "random_forest_optimized"
            elif "SVM" in model_name:
                artifact_path = "svm_optimized"
        
        # Registra novo modelo
        model_uri = f"runs:/{run_id}/{artifact_path}"
        registered_model = mlflow.register_model(
            model_uri=model_uri, 
            name=model_registry_name,
            tags={
                "model_type": model_name,
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "f1_score": str(test_f1_score),
                "promoted": "false"
            }
        )
        
        version_number = registered_model.version
        
        # Define alias 'candidate' para todos os novos modelos
        client.set_registered_model_alias(
            name=model_registry_name,
            alias="candidate",
            version=version_number
        )
        
        # Verifica se deve promover para produção
        if test_f1_score > production_f1:
            # Promove para champion (produção)
            client.set_registered_model_alias(
                name=model_registry_name,
                alias="champion",
                version=version_number
            )
            
            # Atualiza tags
            client.set_model_version_tag(
                name=model_registry_name,
                version=version_number,
                key="promoted",
                value="true"
            )
            
            client.set_model_version_tag(
                name=model_registry_name,
                version=version_number,
                key="promotion_date",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Se havia modelo anterior, move para 'previous'
            if current_champion:
                client.set_registered_model_alias(
                    name=model_registry_name,
                    alias="previous",
                    version=current_champion.version
                )
            
            logger.info(f"🎉 Modelo {model_name} PROMOVIDO para produção!")
            logger.info(f"   Novo F1-Score: {test_f1_score:.4f} > Anterior: {production_f1:.4f}")
            return True
        else:
            logger.info(f"⚠️ Modelo {model_name} NÃO promovido")
            logger.info(f"   F1-Score: {test_f1_score:.4f} <= Produção: {production_f1:.4f}")
            return False
            
    except Exception as e:
        logger.error(f"Erro no registro inteligente: {e}")
        return False

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calcula métricas completas de avaliação
    """
    return {
        'test_accuracy': accuracy_score(y_true, y_pred),
        'test_precision': precision_score(y_true, y_pred, average='weighted'),
        'test_recall': recall_score(y_true, y_pred, average='weighted'),
        'test_f1_score': f1_score(y_true, y_pred, average='weighted'),
        'test_roc_auc': roc_auc_score(y_true, y_proba),
        'test_precision_class_1': precision_score(y_true, y_pred, pos_label=1),
        'test_recall_class_1': recall_score(y_true, y_pred, pos_label=1),
        'test_f1_score_class_1': f1_score(y_true, y_pred, pos_label=1),
    }

def evaluate_model_with_mlflow(model, X_test, y_test, model_name, run_id):
    """
    Avaliação completa usando MLflow 3.0 Evaluation API
    """
    try:
        # Cria dataset de avaliação
        eval_data = pd.DataFrame(X_test)
        eval_data['label'] = y_test
        
        # URI do modelo
        model_uri = f"runs:/{run_id}/model"
        
        # Avaliação completa com MLflow 3.0
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
            custom_metrics=None,
            extra_metrics=None,
            custom_artifacts=None,
            validation_thresholds=None,
            baseline_model=None
        )
        
        logger.info(f"✅ Avaliação MLflow 3.0 completa para {model_name}")
        logger.info(f"   Métricas automáticas capturadas: {len(result.metrics)}")
        
        return result
        
    except Exception as e:
        logger.warning(f"Avaliação MLflow falhou: {e}")
        return None

def train_random_forest(X_train, X_test, y_train, y_test, class_weights, feature_names):
    """
    Treina modelo Random Forest com MLflow 3.0 autologging
    """
    logger.info("Treinando Random Forest...")
    
    # MLflow 3.0: Usar run com nome descritivo e tags organizacionais
    with mlflow.start_run(
        run_name=f"random_forest_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        nested=True
    ) as run:
        
        # Tags para organização (MLflow 3.0 best practices)
        mlflow.set_tags({
            "model_type": "ensemble",
            "algorithm": "random_forest",
            "dataset_version": "v1.0",
            "feature_engineering": "engineered_features",
            "purpose": "predictive_maintenance",
            "team": "ml_ops"
        })
        
        # Parâmetros do modelo
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': class_weights,  # Penalização proporcional
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parâmetros manualmente (além do autolog)
        mlflow.log_params(params)
        mlflow.log_param("penalization_type", "proportional_to_imbalance")
        mlflow.log_param("penalization_ratio", f"{class_weights[1]/class_weights[0]:.2f}x")
        
        # Treina modelo (autolog captura automaticamente)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas manuais (além do autolog)
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        
        # Cria e registra gráficos adicionais
        importance_plot, importance_df = create_feature_importance_plot(model, feature_names, "Random Forest")
        confusion_plot = create_confusion_matrix_plot(y_test, y_pred, "Random Forest")
        roc_plot = create_roc_curve_plot(y_test, y_proba, "Random Forest")
        
        # Log artefatos
        if importance_plot:
            mlflow.log_artifact(importance_plot, "plots")
        mlflow.log_artifact(confusion_plot, "plots")
        mlflow.log_artifact(roc_plot, "plots")
        
        # Log importâncias como JSON
        importance_json_path = None
        if importance_df is not None:
            importance_dict = importance_df.set_index('feature')['importance'].to_dict()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(importance_dict, tmp_file, indent=2)
                importance_json_path = tmp_file.name
            mlflow.log_artifact(importance_json_path, "metrics")
        
        # Log relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(report, tmp_file, indent=2)
            report_json_path = tmp_file.name
        mlflow.log_artifact(report_json_path, "metrics")
        
        # Avaliação com MLflow 3.0 API
        evaluation_result = evaluate_model_with_mlflow(model, X_test, y_test, "Random Forest", run.info.run_id)
        
        # Registro inteligente no Model Registry com aliases
        promoted = register_model_with_intelligent_promotion(
            model, "Random Forest", metrics['test_f1_score'], run.info.run_id, "random_forest"
        )
        
        # Limpa arquivos temporários
        temp_files = [confusion_plot, roc_plot, report_json_path]
        if importance_plot:
            temp_files.append(importance_plot)
        if importance_json_path:
            temp_files.append(importance_json_path)
        
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        
        return model, metrics, promoted

def train_svm(X_train, X_test, y_train, y_test, class_weights, feature_names):
    """
    Treina modelo SVM com MLflow 3.0
    """
    logger.info("Treinando SVM...")
    
    with mlflow.start_run(
        run_name=f"svm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        nested=True
    ) as run:
        
        # Tags organizacionais
        mlflow.set_tags({
            "model_type": "kernel_method",
            "algorithm": "svm",
            "dataset_version": "v1.0",
            "feature_engineering": "engineered_features",
            "purpose": "predictive_maintenance",
            "team": "ml_ops"
        })
        
        # Parâmetros do modelo
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': class_weights,  # Penalização proporcional
            'random_state': 42,
            'probability': True  # Para obter probabilidades
        }
        
        # Log parâmetros
        mlflow.log_params(params)
        mlflow.log_param("penalization_type", "proportional_to_imbalance")
        
        # Treina modelo
        model = SVC(**params)
        model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        
        # Cria e registra gráficos (SVM não tem feature_importances_)
        confusion_plot = create_confusion_matrix_plot(y_test, y_pred, "SVM")
        roc_plot = create_roc_curve_plot(y_test, y_proba, "SVM")
        
        # Log artefatos
        mlflow.log_artifact(confusion_plot, "plots")
        mlflow.log_artifact(roc_plot, "plots")
        
        # Log relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(report, tmp_file, indent=2)
            report_json_path = tmp_file.name
        mlflow.log_artifact(report_json_path, "metrics")
        
        # Salva modelo com signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, 
            "svm", 
            signature=signature,
            input_example=X_train[:5],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        
        # Avaliação MLflow 3.0
        evaluation_result = evaluate_model_with_mlflow(model, X_test, y_test, "SVM", run.info.run_id)
        
        # Registro inteligente no Model Registry
        promoted = register_model_with_intelligent_promotion(
            model, "SVM", metrics['test_f1_score'], run.info.run_id, "svm"
        )
        
        # Limpa arquivos temporários
        for file in [confusion_plot, roc_plot, report_json_path]:
            if os.path.exists(file):
                os.remove(file)
        
        return model, metrics, promoted

def hyperparameter_tuning_random_forest(X_train, y_train, class_weights):
    """
    Hyperparameter tuning para Random Forest usando GridSearchCV
    MLflow 3.0 com max_tuning_runs para controlar nested runs
    """
    logger.info("🔍 Iniciando hyperparameter tuning para Random Forest...")
    
    # Grid de hiperparâmetros simplificado para demonstração rápida
    param_grid = {
        'n_estimators': [50, 100],           # Apenas 2 opções
        'max_depth': [10, 15],               # Apenas 2 opções  
        'min_samples_split': [2, 5],         # Apenas 2 opções
        'min_samples_leaf': [1, 2],          # Apenas 2 opções
        'max_features': ['sqrt'],            # Apenas 1 opção
        'class_weight': [class_weights]      # Usa sempre class_weights calculados
    }
    
    logger.info(f"Grid de parâmetros: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['class_weight'])} combinações")
    
    with mlflow.start_run(
        run_name=f"random_forest_hypertuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        nested=True
    ) as tuning_run:
        
        mlflow.set_tags({
            "model_type": "ensemble",
            "algorithm": "random_forest_gridsearch",
            "tuning_method": "grid_search_cv",
            "dataset_version": "v1.0",
            "purpose": "hyperparameter_optimization",
            "team": "ml_ops"
        })
        
        # Base model para GridSearch
        rf_base = RandomForestClassifier(
            random_state=2024,
            n_jobs=-1
        )
        
        # GridSearchCV com cross-validation estratificado (preserva distribuição de classes)
        stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)
        
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=stratified_cv,  # ✅ CV estratificado para classes desbalanceadas
            scoring='f1',  # Otimizar para F1-score (importante para classes desbalanceadas)
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Log parâmetros do grid search
        mlflow.log_params({
            "cv_folds": 3,
            "cv_strategy": "stratified_k_fold",
            "cv_shuffle": True,
            "scoring_metric": "f1",
            "total_combinations": len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['class_weight']),
            "search_strategy": "grid_search"
        })
        
        # Executa grid search
        logger.info("Executando GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        # Log resultados do melhor modelo
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Calcular também accuracy do CV para o melhor modelo
        from sklearn.model_selection import cross_val_score
        best_cv_accuracy = cross_val_score(
            grid_search.best_estimator_, 
            X_train, 
            y_train, 
            cv=stratified_cv, 
            scoring='accuracy'
        ).mean()
        
        logger.info(f"Melhor F1-Score (CV): {best_score:.4f}")
        logger.info(f"Melhor Accuracy (CV): {best_cv_accuracy:.4f}")
        logger.info(f"Melhores parâmetros: {best_params}")
        
        # Log parâmetros e métricas (com timestamp único para evitar duplicação)
        timestamp_suffix = str(int(time.time() * 1000000))  # Microsegundos
        try:
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                f"rf_hypertuning_cv_f1_score_{timestamp_suffix}": best_score,
                f"rf_hypertuning_cv_mean_{timestamp_suffix}": grid_search.cv_results_['mean_test_score'][grid_search.best_index_],
                f"rf_hypertuning_cv_std_{timestamp_suffix}": grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                f"rf_hypertuning_total_fit_time_{timestamp_suffix}": sum(grid_search.cv_results_['mean_fit_time']),
                f"rf_hypertuning_best_fit_time_{timestamp_suffix}": grid_search.cv_results_['mean_fit_time'][grid_search.best_index_]
            })
        except Exception as e:
            logger.warning(f"Erro ao logar métricas de hypertuning RF: {e}")
        
        # Log top 5 melhores combinações
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_5 = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
        
        logger.info("Top 5 melhores combinações:")
        for idx, row in top_5.iterrows():
            logger.info(f"  F1: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}) - {row['params']}")
        
        # Salva resultados completos como artefato
        with tempfile.NamedTemporaryFile(mode='w', suffix='_grid_search_results.json', delete=False) as tmp_file:
            # Converte class_weights para formato JSON-serializable
            json_params = {}
            for key, value in best_params.items():
                if key == 'class_weight' and isinstance(value, dict):
                    # Converte chaves int64 para int normais
                    json_params[key] = {int(k): float(v) for k, v in value.items()}
                else:
                    json_params[key] = value
            
            results_summary = {
                'best_params': json_params,
                'best_score': float(best_score),
                'cv_results_summary': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': [str(p) for p in grid_search.cv_results_['params']]
                },
                'top_5_combinations': [
                    {
                        'mean_test_score': float(row['mean_test_score']),
                        'std_test_score': float(row['std_test_score']),
                        'params': str(row['params'])
                    }
                    for _, row in top_5.iterrows()
                ]
            }
            json.dump(results_summary, tmp_file, indent=2)
            results_path = tmp_file.name
        
        mlflow.log_artifact(results_path, "hyperparameter_tuning")
        os.remove(results_path)
        
        logger.info("✅ Hyperparameter tuning concluído!")
        
        return grid_search.best_estimator_, best_params, best_score, best_cv_accuracy

def hyperparameter_tuning_svm(X_train, y_train, class_weights):
    """
    Hyperparameter tuning para SVM usando GridSearchCV
    """
    logger.info("🔍 Iniciando hyperparameter tuning para SVM...")
    
    # Grid de hiperparâmetros simplificado para SVM (demonstração rápida)
    param_grid = {
        'C': [1, 10],                        # Apenas 2 opções
        'kernel': ['rbf'],                   # Apenas 1 opção (RBF é geralmente melhor)
        'gamma': ['scale', 'auto'],          # Apenas 2 opções
        'class_weight': [class_weights]      # Usa sempre class_weights calculados
    }
    
    logger.info(f"Grid de parâmetros SVM: {len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma']) * len(param_grid['class_weight'])} combinações")
    
    with mlflow.start_run(
        run_name=f"svm_hypertuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        nested=True
    ) as tuning_run:
        
        mlflow.set_tags({
            "model_type": "kernel_method",
            "algorithm": "svm_gridsearch",
            "tuning_method": "grid_search_cv",
            "dataset_version": "v1.0",
            "purpose": "hyperparameter_optimization",
            "team": "ml_ops"
        })
        
        # Base model para GridSearch
        svm_base = SVC(
            random_state=2024,
            probability=True  # Para ROC-AUC
        )
        
        # GridSearchCV com validação cruzada estratificada
        stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)
        
        grid_search = GridSearchCV(
            estimator=svm_base,
            param_grid=param_grid,
            cv=stratified_cv,  # ✅ CV estratificado para preservar distribuição de classes
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Log parâmetros
        mlflow.log_params({
            "cv_folds": 3,
            "cv_strategy": "stratified_k_fold",
            "cv_shuffle": True,
            "scoring_metric": "f1",
            "total_combinations": len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma']) * len(param_grid['class_weight']),
            "search_strategy": "grid_search"
        })
        
        # Executa grid search
        logger.info("Executando GridSearchCV para SVM...")
        grid_search.fit(X_train, y_train)
        
        # Log resultados
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Calcular também accuracy do CV para o melhor modelo SVM
        best_cv_accuracy = cross_val_score(
            grid_search.best_estimator_, 
            X_train, 
            y_train, 
            cv=stratified_cv, 
            scoring='accuracy'
        ).mean()
        
        logger.info(f"Melhor F1-Score SVM (CV): {best_score:.4f}")
        logger.info(f"Melhor Accuracy SVM (CV): {best_cv_accuracy:.4f}")
        logger.info(f"Melhores parâmetros SVM: {best_params}")
        
        # Log métricas com timestamp único para evitar duplicadas
        timestamp_suffix = str(int(time.time() * 1000000))  # Microsegundos
        try:
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                f"svm_hypertuning_cv_f1_score_{timestamp_suffix}": best_score,
                f"svm_hypertuning_cv_mean_{timestamp_suffix}": grid_search.cv_results_['mean_test_score'][grid_search.best_index_],
                f"svm_hypertuning_cv_std_{timestamp_suffix}": grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                f"svm_hypertuning_total_fit_time_{timestamp_suffix}": sum(grid_search.cv_results_['mean_fit_time']),
                f"svm_hypertuning_best_fit_time_{timestamp_suffix}": grid_search.cv_results_['mean_fit_time'][grid_search.best_index_]
            })
        except Exception as e:
            logger.warning(f"Erro ao logar métricas SVM hypertuning: {e}")
        
        logger.info("✅ Hyperparameter tuning SVM concluído!")
        
        return grid_search.best_estimator_, best_params, best_score, best_cv_accuracy

def preprocess_data(X_train, X_test):
    """
    Preprocessa dados aplicando normalização APENAS após split (evita data leakage)
    """
    logger.info("Preprocessando dados...")
    
    # Aplica StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Salva o scaler treinado localmente (backup)
    os.makedirs('/opt/airflow/models', exist_ok=True)
    scaler_path = '/opt/airflow/models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler treinado salvo localmente")
    
    return X_train_scaled, X_test_scaled, scaler

def main():
    """
    Pipeline principal de treinamento com MLflow 3.0
    """
    try:
        logger.info("Iniciando pipeline de treinamento...")
    
        # Carrega dados
        df = load_features()
        
        # Prepara features e target
        feature_columns = [col for col in df.columns if col != 'target']
        X = df[feature_columns]
        y = df['target']
        
        logger.info(f"Dataset: {len(df)} amostras, {len(feature_columns)} features")
        logger.info(f"Distribuição do target: {dict(y.value_counts())}")
        
        # Calcula pesos para penalização proporcional
        class_weights = calculate_class_weights(y)
        
        # Split treino/teste estratificado com embaralhamento
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2024, stratify=y, shuffle=True
        )
        
        logger.info(f"Treino: {len(X_train)} amostras")
        logger.info(f"Teste: {len(X_test)} amostras")
        
        # Preprocessa dados (normalização após split)
        X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
        
        # Experiment tracking com MLflow 3.0
        with mlflow.start_run(
            run_name=f"predictive_maintenance_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as parent_run:
            
            # Tags do experimento principal
            mlflow.set_tags({
                "experiment_type": "model_comparison",
                "dataset": "ai4i_2020_type_l",
                "dataset_size": str(len(df)),
                "split_strategy": "stratified_shuffled",
                "preprocessing": "standard_scaler",
                "validation_type": "holdout",
                "team": "ml_ops",
                "project": "predictive_maintenance"
            })
            
            # Log informações do dataset
            mlflow.log_params({
                "total_samples": len(df),
                "n_features": len(feature_columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "positive_class_ratio": float(y.sum() / len(y)),
                "class_weight_0": class_weights[0],
                "class_weight_1": class_weights[1]
            })
            
            # Salva o scaler no MLflow como artefato para uso na inferência
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                joblib.dump(scaler, tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "preprocessors/scaler.pkl")
                scaler_mlflow_path = tmp_file.name
            
            # Salva informações do scaler como parâmetros
            mlflow.log_params({
                "scaler_type": "StandardScaler",
                "n_features_in": scaler.n_features_in_,
                "feature_names_in": list(X_train.columns) if hasattr(X_train, 'columns') else None,
                "scaler_mean": list(scaler.mean_),
                "scaler_scale": list(scaler.scale_)
            })
            
            logger.info("Scaler salvo no MLflow")
            
            # Remove arquivo temporário
            if os.path.exists(scaler_mlflow_path):
                os.remove(scaler_mlflow_path)
            
            # Treina diferentes modelos com hyperparameter tuning
            results = {}
            
            # Hyperparameter Tuning Random Forest
            logger.info("\n" + "="*50)
            logger.info("🎯 HYPERPARAMETER TUNING - RANDOM FOREST")
            logger.info("="*50)
            
            best_rf_model, best_rf_params, best_rf_cv_score, best_rf_cv_accuracy = hyperparameter_tuning_random_forest(
                X_train_scaled, y_train, class_weights
            )
            
            # Avaliação do modelo otimizado no conjunto de teste
            logger.info("\n" + "="*50)
            logger.info("📊 AVALIAÇÃO FINAL - RANDOM FOREST (Conjunto de Teste)")
            logger.info("="*50)
            
            # O GridSearchCV já treinou o modelo final com os melhores parâmetros
            # Agora calculamos métricas completas: CV, Treino e Teste
            
            # 1. Métricas no conjunto de TREINO (verifica se o modelo decorou)
            y_pred_rf_train = best_rf_model.predict(X_train_scaled)
            y_proba_rf_train = best_rf_model.predict_proba(X_train_scaled)[:, 1]
            rf_train_metrics = calculate_metrics(y_train, y_pred_rf_train, y_proba_rf_train)
            
            # 2. Métricas no conjunto de TESTE ("virgem")
            y_pred_rf = best_rf_model.predict(X_test_scaled)
            y_proba_rf = best_rf_model.predict_proba(X_test_scaled)[:, 1]
            rf_test_metrics = calculate_metrics(y_test, y_pred_rf, y_proba_rf)
            
            # Consolida métricas completas
            rf_metrics = {
                **rf_test_metrics,  # Métricas de teste (com prefixo test_)
                'cv_f1_score': best_rf_cv_score,
                'cv_accuracy': best_rf_cv_accuracy,
                'train_f1_score': rf_train_metrics['test_f1_score'],  # Renomeia para train_
                'train_accuracy': rf_train_metrics['test_accuracy']    # Renomeia para train_
            }
            
            logger.info(f"📊 Performance Completa - Random Forest:")
            logger.info(f"   F1-Score (CV):    {best_rf_cv_score:.4f} | Accuracy (CV):    {best_rf_cv_accuracy:.4f}")
            logger.info(f"   F1-Score (Train): {rf_metrics['train_f1_score']:.4f} | Accuracy (Train): {rf_metrics['train_accuracy']:.4f}")
            logger.info(f"   F1-Score (Test):  {rf_metrics['test_f1_score']:.4f} | Accuracy (Test):  {rf_metrics['test_accuracy']:.4f}")
            logger.info(f"   Diferença Train-Test: {abs(rf_metrics['train_f1_score'] - rf_metrics['test_f1_score']):.4f}")
            logger.info(f"📋 Parâmetros ótimos utilizados: {best_rf_params}")
            
            # Registra modelo Random Forest otimizado
            with mlflow.start_run(
                run_name=f"random_forest_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                nested=True
            ) as rf_final_run:
                
                mlflow.set_tags({
                    "model_type": "ensemble_optimized",
                    "algorithm": "random_forest_tuned",
                    "optimization_method": "grid_search_cv",
                    "dataset_version": "v1.0",
                    "purpose": "production_model",
                    "team": "ml_ops"
                })
                
                # Log parâmetros otimizados e métricas
                mlflow.log_params(best_rf_params)
                mlflow.log_metrics(rf_metrics)
                
                # Log modelo otimizado
                signature = infer_signature(X_train_scaled, y_pred_rf)
                mlflow.sklearn.log_model(
                    best_rf_model,
                    "random_forest_optimized",
                    signature=signature,
                    input_example=X_train_scaled[:5],
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                )
                
                # Registro inteligente no Model Registry
                rf_promoted = register_model_with_intelligent_promotion(
                    best_rf_model, "RandomForest_Optimized", rf_metrics['test_f1_score'], rf_final_run.info.run_id, "random_forest_optimized"
                )
            
            results['random_forest_tuned'] = rf_metrics
            
            # Hyperparameter Tuning SVM
            logger.info("\n" + "="*50)
            logger.info("🎯 HYPERPARAMETER TUNING - SVM")
            logger.info("="*50)
            
            best_svm_model, best_svm_params, best_svm_cv_score, best_svm_cv_accuracy = hyperparameter_tuning_svm(
                X_train_scaled, y_train, class_weights
            )
            
            # Avaliação do SVM otimizado no conjunto de teste
            logger.info("\n" + "="*50)
            logger.info("📊 AVALIAÇÃO FINAL - SVM (Conjunto de Teste)")
            logger.info("="*50)
            
            # O GridSearchCV já treinou o modelo SVM final com os melhores parâmetros
            # Agora calculamos métricas completas: CV, Treino e Teste
            
            # 1. Métricas no conjunto de TREINO (verifica se o modelo decorou)
            y_pred_svm_train = best_svm_model.predict(X_train_scaled)
            y_proba_svm_train = best_svm_model.predict_proba(X_train_scaled)[:, 1]
            svm_train_metrics = calculate_metrics(y_train, y_pred_svm_train, y_proba_svm_train)
            
            # 2. Métricas no conjunto de TESTE ("virgem")
            y_pred_svm = best_svm_model.predict(X_test_scaled)
            y_proba_svm = best_svm_model.predict_proba(X_test_scaled)[:, 1]
            svm_test_metrics = calculate_metrics(y_test, y_pred_svm, y_proba_svm)
            
            # Consolida métricas completas
            svm_metrics = {
                **svm_test_metrics,  # Métricas de teste (com prefixo test_)
                'cv_f1_score': best_svm_cv_score,
                'cv_accuracy': best_svm_cv_accuracy,
                'train_f1_score': svm_train_metrics['test_f1_score'],  # Renomeia para train_
                'train_accuracy': svm_train_metrics['test_accuracy']    # Renomeia para train_
            }
            
            logger.info(f"📊 Performance Completa - SVM:")
            logger.info(f"   F1-Score (CV):    {best_svm_cv_score:.4f} | Accuracy (CV):    {best_svm_cv_accuracy:.4f}")
            logger.info(f"   F1-Score (Train): {svm_metrics['train_f1_score']:.4f} | Accuracy (Train): {svm_metrics['train_accuracy']:.4f}")
            logger.info(f"   F1-Score (Test):  {svm_metrics['test_f1_score']:.4f} | Accuracy (Test):  {svm_metrics['test_accuracy']:.4f}")
            logger.info(f"   Diferença Train-Test: {abs(svm_metrics['train_f1_score'] - svm_metrics['test_f1_score']):.4f}")
            logger.info(f"📋 Parâmetros ótimos utilizados: {best_svm_params}")
            
            # Registra modelo SVM otimizado
            with mlflow.start_run(
                run_name=f"svm_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                nested=True
            ) as svm_final_run:
                
                mlflow.set_tags({
                    "model_type": "kernel_method_optimized",
                    "algorithm": "svm_tuned",
                    "optimization_method": "grid_search_cv",
                    "dataset_version": "v1.0",
                    "purpose": "production_model",
                    "team": "ml_ops"
                })
                
                # Log parâmetros otimizados e métricas
                mlflow.log_params(best_svm_params)
                mlflow.log_metrics(svm_metrics)
                
                # Log modelo otimizado
                signature = infer_signature(X_train_scaled, y_pred_svm)
                mlflow.sklearn.log_model(
                    best_svm_model,
                    "svm_optimized",
                    signature=signature,
                    input_example=X_train_scaled[:5],
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                )
                
                # Registro inteligente no Model Registry
                svm_promoted = register_model_with_intelligent_promotion(
                    best_svm_model, "SVM_Optimized", svm_metrics['test_f1_score'], svm_final_run.info.run_id, "svm_optimized"
                )
            
            results['svm_tuned'] = svm_metrics
            
            # Summary dos resultados
            logger.info("\n" + "="*50)
            logger.info("📊 RESUMO DOS RESULTADOS - HYPERPARAMETER TUNING:")
            logger.info("="*50)
            
            for model_name, metrics in results.items():
                logger.info(f"\n🤖 {model_name.upper()}:")
                logger.info(f"   F1-Score (CV):    {metrics['cv_f1_score']:.4f} | Accuracy (CV):    {metrics['cv_accuracy']:.4f}")
                logger.info(f"   F1-Score (Train): {metrics['train_f1_score']:.4f} | Accuracy (Train): {metrics['train_accuracy']:.4f}")
                logger.info(f"   F1-Score (Test):  {metrics['test_f1_score']:.4f} | Accuracy (Test):  {metrics['test_accuracy']:.4f}")
                logger.info(f"   Train-Test Diff:  {abs(metrics['train_f1_score'] - metrics['test_f1_score']):.4f}")
                logger.info(f"   ROC-AUC:          {metrics['test_roc_auc']:.4f}")
                logger.info(f"   F1 Falhas:        {metrics['test_f1_score_class_1']:.4f}")
            
            # Identifica melhor modelo
            best_model = max(results.keys(), key=lambda k: results[k]['test_f1_score'])
            best_f1_test = results[best_model]['test_f1_score']
            best_f1_train = results[best_model]['train_f1_score']
            best_f1_cv = results[best_model]['cv_f1_score']
            best_acc_test = results[best_model]['test_accuracy']
            best_acc_train = results[best_model]['train_accuracy']
            best_acc_cv = results[best_model]['cv_accuracy']
            
            logger.info(f"\n🏆 MELHOR MODELO: {best_model.upper()}")
            logger.info(f"   F1-Score (CV):    {best_f1_cv:.4f} | Accuracy (CV):    {best_acc_cv:.4f}")
            logger.info(f"   F1-Score (Train): {best_f1_train:.4f} | Accuracy (Train): {best_acc_train:.4f}")
            logger.info(f"   F1-Score (Test):  {best_f1_test:.4f} | Accuracy (Test):  {best_acc_test:.4f}")
            logger.info(f"   Diferença Train-Test: {abs(best_f1_train - best_f1_test):.4f}")
            
            # Verifica overfitting (Train vs Test é a métrica correta)
            train_test_diff = abs(best_f1_train - best_f1_test)
            if train_test_diff > 0.05:
                logger.warning(f"⚠️  Possível overfitting detectado (Train-Test diff: {train_test_diff:.4f})")
            else:
                logger.info(f"Modelo generaliza bem (Train-Test diff: {train_test_diff:.4f})")
                
            # Log métricas finais do experimento
            mlflow.log_metrics({
                "best_model_f1_test": best_f1_test,
                "best_model_f1_train": best_f1_train,
                "best_model_f1_cv": best_f1_cv,
                "best_model_accuracy_test": best_acc_test,
                "best_model_accuracy_train": best_acc_train,
                "best_model_accuracy_cv": best_acc_cv,
                "train_test_difference": train_test_diff,
                "models_trained": len(results),
                "hyperparameter_tuning_enabled": 1
            })
            
            mlflow.log_param("best_model", best_model)
            mlflow.log_param("tuning_method", "grid_search_cv")
            
            logger.info("\n✅ Pipeline de treinamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na pipeline de treinamento: {e}")
        raise

if __name__ == "__main__":
    main() 