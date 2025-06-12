#!/bin/bash
set -e

# Função para aguardar o PostgreSQL
wait_for_postgres() {
    echo "Aguardando PostgreSQL..."
    while ! nc -z postgres 5432; do
        echo "PostgreSQL não está pronto - aguardando..."
        sleep 2
    done
    echo "PostgreSQL está pronto!"
}

# Função para aguardar o Redis
wait_for_redis() {
    echo "Aguardando Redis..."
    while ! nc -z redis 6379; do
        echo "Redis não está pronto - aguardando..."
        sleep 2
    done
    echo "Redis está pronto!"
}

# Função para inicializar o Airflow
init_airflow() {
    echo "Inicializando banco de dados do Airflow..."
    airflow db init
    
    echo "Criando usuário admin..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || echo "Usuário já existe"
}

# Aguarda serviços dependentes
wait_for_postgres

# Para worker, também aguarda Redis
if [ "$1" = "celery" ] && [ "$2" = "worker" ]; then
    wait_for_redis
fi

# Inicializa apenas se for o webserver
if [ "$1" = "webserver" ]; then
    init_airflow
fi

# Limpa arquivo PID se for worker
if [ "$1" = "celery" ] && [ "$2" = "worker" ]; then
    echo "Limpando arquivo PID do worker..."
    rm -f /opt/airflow/airflow-worker.pid
fi

# Executa o comando do Airflow
exec airflow "$@"