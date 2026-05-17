// PM2 Process Manager Configuration — CTH621 Pipeline
// Dùng cho long-running batch jobs trên Linux server
// Cài PM2: npm install -g pm2
// Khởi chạy: pm2 start ecosystem.config.js
// Xem log:   pm2 logs cth621-pipeline
// Dừng:      pm2 stop cth621-pipeline

module.exports = {
  apps: [
    {
      name: "cth621-pipeline",
      script: "src/presentation/run_pipeline.py",
      interpreter: "python3",
      // Thay --task và --dataset theo nhu cầu khi deploy
      args: "--task all --dataset student_performance --config configs/params.yaml",
      cwd: "./",
      autorestart: false,         // Không tự restart vì là batch job (chạy 1 lần)
      watch: false,
      max_memory_restart: "4G",
      log_file: "logs/pm2_combined.log",
      out_file: "logs/pm2_out.log",
      error_file: "logs/pm2_err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: {
        PYTHONUNBUFFERED: "1",    // Đảm bảo log hiện thời gian thực
        PYTHONPATH: "."
      }
    },
    {
      // Instance chạy MLflow tracking server (nếu dùng MLflow)
      name: "cth621-mlflow",
      script: "mlflow",
      interpreter: "python3",
      args: "server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db",
      autorestart: true,
      watch: false,
      out_file: "logs/mlflow_out.log",
      error_file: "logs/mlflow_err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    }
  ]
};
