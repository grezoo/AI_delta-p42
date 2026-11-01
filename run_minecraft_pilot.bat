@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title ΔP-42 Sovereign Cognitive Core — Minecraft Pilot Runner

echo ================================================================================
echo   ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — MINECRAFT EMBODIED PILOT
echo ================================================================================
echo.
echo 1. Indítsd el a Minecraftot (TLauncher)!
echo 2. Hozz létre vagy nyiss meg egy egyjátékos (Singleplayer) világot!
echo 3. Nyomj ESC-et, majd kattints: "Megnyitás a helyi hálózaton" (Open to LAN)!
echo.
echo Tipp: Ha üresen hagyod és csak ENTER-t nyomsz, a rendszer AUTOMATIKUSAN
echo       megkeresi és felderíti a nyitott világot a hálózaton!
echo.
set /p MC_PORT="Portszám (üresen = AUTOMATIKUS felderítés): "

echo.
echo [1/2] Webes telemetria cockpit megnyitása (http://localhost:8080)...
start http://localhost:8080

cd /d "C:\GOOGLE PROJEKTEK\ai_nursery"

if "%MC_PORT%"=="" (
    echo [2/2] ΔP-42 kognitív mag indítása AUTOMATIKUS LAN felderítéssel...
    python nursery/minecraft_bridge.py
) else (
    echo [2/2] ΔP-42 kognitív mag indítása a megadott %MC_PORT% porton...
    python nursery/minecraft_bridge.py --port %MC_PORT%
)

pause
