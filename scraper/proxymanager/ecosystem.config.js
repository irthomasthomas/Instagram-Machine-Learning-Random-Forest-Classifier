module.exports = {
  apps : [{
    name: 'proxies',
    cmd: 'gotproxies.py',
    interpreter: 'python3',
    args: '-u',
    // Options reference: https://pm2.keymetrics.io/docs/usage/application-declaration/
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '500M'
  }]
};
