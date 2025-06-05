// Verify if ports are in use
const net = require('net');

function checkPort(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.log(`Puerto ${port} está en uso.`);
        resolve(false);
      } else {
        console.log(`Error comprobando puerto ${port}: ${err.code}`);
        resolve(false);
      }
    });
    
    server.once('listening', () => {
      server.close();
      console.log(`Puerto ${port} está disponible.`);
      resolve(true);
    });
    
    server.listen(port);
  });
}

async function checkPorts() {
  const ports = [3000, 3443, 8000];
  
  for (const port of ports) {
    await checkPort(port);
  }
}

checkPorts();
