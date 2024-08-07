# Menggunakan image tensorflow/serving terbaru
FROM tensorflow/serving:latest

# Menyalin model ke direktori /models dalam container
COPY ./serving_model_dir /models
COPY ./monitoring/prometheus.config /model_config/prometheus.config

# Mendefinisikan environment variables
ENV MODEL_NAME=spam-detection-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

# Membuat skrip entrypoint untuk menjalankan tensorflow_model_server
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh