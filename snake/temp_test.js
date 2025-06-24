
        // Get canvas and context
        const canvas = document.getElementById('arena-canvas');
        const ctx = canvas.getContext('2d');
        const log = document.getElementById('battle-log');
        
        // Battle state
        let battleActive = false;
        let animationId = null;
        
        // Snake names and colors
        const snakeNames = [
            'Crimson Viper', 'Azure Python', 'Golden Cobra', 'Shadow Mamba',
            'Frost Serpent', 'Ember Snake', 'Storm Adder', 'Void Boa',
            'Neon Rattler', 'Crystal Anaconda', 'Phantom Asp', 'Thunder Viper'
        ];
        
        const snakeColors = [
            '#ff0000', '#0066ff', '#ffaa00', '#8800ff',
            '#00ff88', '#ff0088', '#88ff00', '#ff8800',
            '#0088ff', '#ff4400', '#44ff00', '#ff0044'
        ];
        
        const skills = [
            'Lightning Strike', 'Iron Scales', 'Speed Boost', 'Venom Bite',
            'Shield Barrier', 'Rage Mode', 'Stealth', 'Power Core',
            'Quick Dodge', 'Mighty Coil', 'Heat Vision', 'Battle Fury'
        ];
        
        // Logging function
        function addLog(message, color = '#00ffff') {
            const time = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.style.color = color;
            logEntry.innerHTML = `<span style="color: #666;">[${time}]</span> ${message}`;
            log.appendChild(logEntry);
            log.scrollTop = log.scrollHeight;
            
            if (log.children.length > 50) {
                log.removeChild(log.firstChild);
            }
        }
        
        // Generate random snake
        function generateSnake() {
            const name = snakeNames[Math.floor(Math.random() * snakeNames.length)];
            const color = snakeColors[Math.floor(Math.random() * snakeColors.length)];
            const snakeSkills = [];
            
            // Random skills (1-3)
            const skillCount = Math.floor(Math.random() * 3) + 1;
            for (let i = 0; i < skillCount; i++) {
                const skill = skills[Math.floor(Math.random() * skills.length)];
                if (!snakeSkills.includes(skill)) {
                    snakeSkills.push(skill);
                }
            }
            
            return {
                name: name,
                color: color,
                skills: snakeSkills,
                health: 100,
                x: 0,
                y: 0,
                size: Math.random() * 15 + 20,
                speed: Math.random() * 3 + 2
            };
        }
        
        // Draw arena
        function drawArena() {
            // Clear canvas
            ctx.fillStyle = '#001122';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw border
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 3;
            ctx.strokeRect(5, 5, canvas.width - 10, canvas.height - 10);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
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
            ctx.beginPath();
            ctx.arc(x, y, snake.size, 0, Math.PI * 2);
            ctx.fill();
            
            // Snake border
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Eyes
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(x - 8, y - 8, 3, 0, Math.PI * 2);
            ctx.arc(x + 8, y - 8, 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Name
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(snake.name, x, y - snake.size - 15);
            
            // Skills
            ctx.font = '10px Arial';
            ctx.fillText(`Skills: ${snake.skills.join(', ')}`, x, y + snake.size + 20);
        }
        
        // Start instant battle        function startInstantBattle() {
            addLog('🔥 Starting instant battle!', '#ff9900');
            
            const snake1 = generateSnake();
            const snake2 = generateSnake();
            
            addLog(`Fighter 1: ${snake1.name} (${snake1.color})`, snake1.color);
            addLog(`Skills: ${snake1.skills.join(', ')}`, '#ffff00');
            addLog(`Fighter 2: ${snake2.name} (${snake2.color})`, snake2.color);
            addLog(`Skills: ${snake2.skills.join(', ')}`, '#ffff00');
            
            // Draw initial battle setup
            drawArena();
            
            // Position snakes
            const snake1X = 200;
            const snake1Y = canvas.height / 2;
            const snake2X = canvas.width - 200;
            const snake2Y = canvas.height / 2;
            
            drawSnake(snake1, snake1X, snake1Y);
            drawSnake(snake2, snake2X, snake2Y);
            
            // VS text
            ctx.fillStyle = '#ffff00';
            ctx.font = 'bold 48px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('VS', canvas.width / 2, canvas.height / 2);
            
            // Start countdown before battle
            startCountdown(snake1, snake2, snake1X, snake1Y, snake2X, snake2Y);
        }
        
        // Countdown function
        function startCountdown(snake1, snake2, x1, y1, x2, y2) {
            let countdown = 3;
            
            function showCountdown() {
                // Redraw arena and snakes
                drawArena();
                drawSnake(snake1, x1, y1);
                drawSnake(snake2, x2, y2);
                
                // Draw countdown number
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                if (countdown > 0) {
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
                    // Show "FIGHT!"
                    ctx.fillStyle = '#00ff00';
                    ctx.font = 'bold 80px Arial';
                    ctx.strokeText('FIGHT!', canvas.width / 2, canvas.height / 2);
                    ctx.fillText('FIGHT!', canvas.width / 2, canvas.height / 2);
                    
                    addLog('⚔️ FIGHT!', '#00ff00');
                    
                    // Start battle after 500ms
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
            const maxFrames = 300; // 5 seconds at 60fps
            
            function animate() {
                if (!battleActive || frame >= maxFrames) {
                    endBattle(snake1, snake2);
                    return;
                }
                
                // Clear and redraw arena
                drawArena();
                
                // Animate snakes moving towards each other
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
            
            // Determine winner
            const winner = Math.random() > 0.5 ? snake1 : snake2;
            const loser = winner === snake1 ? snake2 : snake1;
            
            addLog(`🏆 WINNER: ${winner.name}!`, '#00ff00');
            addLog(`💀 ${loser.name} is defeated!`, '#ff0000');
            
            // Draw victory screen
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
            
            // Auto-reset after 3 seconds
            setTimeout(() => {
                resetArena();
            }, 3000);
        }
        
        // Start tournament
        function startTournament() {
            addLog('🏆 Starting 4-snake tournament!', '#ffff00');
            
            const fighters = [];
            for (let i = 0; i < 4; i++) {
                fighters.push(generateSnake());
            }
            
            fighters.forEach((fighter, index) => {
                addLog(`Fighter ${index + 1}: ${fighter.name}`, fighter.color);
            });
            
            drawTournamentBracket(fighters);
        }
        
        // Draw tournament bracket
        function drawTournamentBracket(fighters) {
            drawArena();
            
            ctx.fillStyle = '#ffff00';
            ctx.font = 'bold 32px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🏆 TOURNAMENT BRACKET', canvas.width / 2, 60);
            
            // Draw fighters
            const positions = [
                { x: 150, y: 150 },
                { x: 150, y: 250 },
                { x: 150, y: 350 },
                { x: 150, y: 450 }
            ];
            
            fighters.forEach((fighter, index) => {
                const pos = positions[index];
                
                // Fighter circle
                ctx.fillStyle = fighter.color;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 25, 0, Math.PI * 2);
                ctx.fill();
                
                // Fighter name
                ctx.fillStyle = '#ffffff';
                ctx.font = '14px Arial';
                ctx.textAlign = 'left';
                ctx.fillText(fighter.name, pos.x + 35, pos.y + 5);
            });
            
            // Draw bracket lines
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 2;
            
            // Semi-finals
            ctx.beginPath();
            ctx.moveTo(200, 150);
            ctx.lineTo(350, 200);
            ctx.moveTo(200, 250);
            ctx.lineTo(350, 200);
            ctx.moveTo(200, 350);
            ctx.lineTo(350, 400);
            ctx.moveTo(200, 450);
            ctx.lineTo(350, 400);
            ctx.stroke();
            
            // Finals
            ctx.beginPath();
            ctx.moveTo(350, 200);
            ctx.lineTo(500, 300);
            ctx.moveTo(350, 400);
            ctx.lineTo(500, 300);
            ctx.stroke();
            
            // Trophy
            ctx.fillStyle = '#ffff00';
            ctx.font = '48px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🏆', 650, 320);
            
            addLog('Tournament bracket created!', '#00ff00');
        }
        
        // Reset arena
        function resetArena() {
            battleActive = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            
            drawArena();
            
            ctx.fillStyle = '#00ffff';
            ctx.font = 'bold 24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('🐍 Elite Snake Battle Arena', canvas.width / 2, canvas.height / 2 - 20);
            
            ctx.font = '16px Arial';
            ctx.fillText('Ready for Epic Battles!', canvas.width / 2, canvas.height / 2 + 20);
            
            addLog('Arena reset - ready for battle!', '#00ff00');
        }
        
        // Initialize
        window.addEventListener('load', function() {
            addLog('🎮 Snake Arena loaded successfully!', '#00ff00');
            resetArena();
        });
        
        addLog('🐍 Welcome to the Elite Snake Arena!', '#00ffff');
    