
        // Get canvas and context
        const canvas = document.getElementById('arena-canvas');
        const ctx = canvas.getContext('2d');
        const log = document.getElementById('battle-log');
        
        // Battle state
        let battleActive = false;
        let animationId = null;
        
        // Add log entry
        function addLog(message, color = '#00ffaa') {
            const entry = document.createElement('div');
            entry.style.color = color;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        // Generate random snake
        function generateSnake() {
            const names = ['Viper', 'Cobra', 'Python', 'Mamba', 'Adder', 'Boa', 'Anaconda', 'Rattler'];
            const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ff8800', '#8800ff'];
            const skills = ['Lightning Strike', 'Venom Bite', 'Coil Crush', 'Speed Boost', 'Iron Scales', 'Heat Vision'];
            
            return {
                name: names[Math.floor(Math.random() * names.length)] + '-' + Math.floor(Math.random() * 1000),
                color: colors[Math.floor(Math.random() * colors.length)],
                skills: [skills[Math.floor(Math.random() * skills.length)], skills[Math.floor(Math.random() * skills.length)]],
                power: Math.floor(Math.random() * 100) + 50
            };
        }
        
        // Draw arena background
        function drawArena() {
            // Clear canvas
            ctx.fillStyle = 'rgba(0, 26, 26, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Arena grid
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 50) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 50) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
        }
        
        // Draw snake
        function drawSnake(snake, x, y) {
            // Snake body
            ctx.fillStyle = snake.color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            
            // Main body
            ctx.beginPath();
            ctx.arc(x, y, 30, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            
            // Eyes
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x - 10, y - 10, 5, 0, Math.PI * 2);
            ctx.arc(x + 10, y - 10, 5, 0, Math.PI * 2);
            ctx.fill();
            
            // Pupils
            ctx.fillStyle = '#000000';
            ctx.beginPath();
            ctx.arc(x - 10, y - 10, 2, 0, Math.PI * 2);
            ctx.arc(x + 10, y - 10, 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Name
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(snake.name, x, y + 50);
        }
        
        // Reset arena
        function resetArena() {
            battleActive = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            drawArena();
            addLog('🔄 Arena reset!', '#00ffff');
        }
        
        // Start instant battle with countdown
        function startInstantBattle() {
            if (battleActive) {
                addLog('⚠️ Battle already in progress!', '#ff9900');
                return;
            }
            
            addLog('🔥 Starting instant battle!', '#ff9900');
            
            const snake1 = generateSnake();
            const snake2 = generateSnake();
            
            addLog(`Fighter 1: ${snake1.name} (Power: ${snake1.power})`, snake1.color);
            addLog(`Skills: ${snake1.skills.join(', ')}`, '#ffff00');
            addLog(`Fighter 2: ${snake2.name} (Power: ${snake2.power})`, snake2.color);
            addLog(`Skills: ${snake2.skills.join(', ')}`, '#ffff00');
            
            // Position snakes
            const snake1X = 200;
            const snake1Y = canvas.height / 2;
            const snake2X = canvas.width - 200;
            const snake2Y = canvas.height / 2;
            
            // Start countdown
            startCountdown(snake1, snake2, snake1X, snake1Y, snake2X, snake2Y);
        }
        
        // Countdown function
        function startCountdown(snake1, snake2, x1, y1, x2, y2) {
            let countdown = 3;
            
            function showCountdown() {
                // Draw arena and snakes
                drawArena();
                drawSnake(snake1, x1, y1);
                drawSnake(snake2, x2, y2);
                
                // Semi-transparent overlay
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                if (countdown > 0) {
                    // Countdown number
                    ctx.fillStyle = '#ff0000';
                    ctx.font = 'bold 120px Arial';
                    ctx.textAlign = 'center';
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 4;
                    ctx.strokeText(countdown.toString(), canvas.width / 2, canvas.height / 2);
                    ctx.fillText(countdown.toString(), canvas.width / 2, canvas.height / 2);
                    
                    addLog(`⏰ ${countdown}...`, '#ff9900');
                    countdown--;
                    setTimeout(showCountdown, 1000);
                } else {
                    // FIGHT!
                    ctx.fillStyle = '#00ff00';
                    ctx.font = 'bold 80px Arial';
                    ctx.strokeText('FIGHT!', canvas.width / 2, canvas.height / 2);
                    ctx.fillText('FIGHT!', canvas.width / 2, canvas.height / 2);
                    
                    addLog('⚔️ FIGHT!', '#00ff00');
                    
                    // Start battle
                    setTimeout(() => {
                        battleActive = true;
                        animateBattle(snake1, snake2, x1, y1, x2, y2);
                    }, 500);
                }
            }
            
            showCountdown();
        }
        
        // Animate battle
        function animateBattle(snake1, snake2, x1, y1, x2, y2) {
            let frame = 0;
            const maxFrames = 300; // 5 seconds
            
            function animate() {
                if (!battleActive || frame >= maxFrames) {
                    endBattle(snake1, snake2);
                    return;
                }
                
                // Draw arena
                drawArena();
                
                // Animate snakes
                const progress = frame / maxFrames;
                const moveDistance = 50;
                
                const currentX1 = x1 + Math.sin(frame * 0.1) * 30 + progress * moveDistance;
                const currentY1 = y1 + Math.cos(frame * 0.15) * 20;
                const currentX2 = x2 - Math.sin(frame * 0.1) * 30 - progress * moveDistance;
                const currentY2 = y2 + Math.cos(frame * 0.15 + Math.PI) * 20;
                
                drawSnake(snake1, currentX1, currentY1);
                drawSnake(snake2, currentX2, currentY2);
                
                // Battle effects
                if (frame % 30 === 0) {
                    addLog(`💥 ${Math.random() > 0.5 ? snake1.name : snake2.name} attacks!`, '#ff6600');
                }
                
                // Progress bar
                ctx.fillStyle = '#333333';
                ctx.fillRect(50, canvas.height - 30, canvas.width - 100, 15);
                ctx.fillStyle = '#00ff00';
                ctx.fillRect(50, canvas.height - 30, (canvas.width - 100) * progress, 15);
                
                frame++;
                animationId = requestAnimationFrame(animate);
            }
            
            animate();
        }
        
        // End battle
        function endBattle(snake1, snake2) {
            battleActive = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            
            // Determine winner based on power
            const winner = snake1.power > snake2.power ? snake1 : snake2;
            const loser = winner === snake1 ? snake2 : snake1;
            
            addLog(`🏆 WINNER: ${winner.name}! (Power: ${winner.power})`, '#00ff00');
            addLog(`💀 ${loser.name} is defeated! (Power: ${loser.power})`, '#ff0000');
            
            // Victory screen
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#ffff00';
            ctx.font = 'bold 36px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🏆 VICTORY! 🏆', canvas.width / 2, canvas.height / 2 - 60);
            
            ctx.fillStyle = winner.color;
            ctx.font = 'bold 28px Arial';
            ctx.fillText(winner.name, canvas.width / 2, canvas.height / 2);
            
            ctx.fillStyle = '#ffffff';
            ctx.font = '18px Arial';
            ctx.fillText(`Defeated ${loser.name} in epic combat!`, canvas.width / 2, canvas.height / 2 + 40);
            
            // Auto-reset
            setTimeout(() => {
                resetArena();
            }, 3000);
        }
        
        // Tournament placeholder
        function startTournament() {
            addLog('🏆 Tournament mode coming soon!', '#ffff00');
        }
        
        // Initialize
        drawArena();
        addLog('🎮 Arena ready! Click INSTANT BATTLE to start!', '#00ffff');
        
    