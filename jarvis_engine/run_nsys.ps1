$n = (Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3' -Recurse -Filter nsys.exe | Select-Object -First 1).FullName
Set-Location $PSScriptRoot\..
& $n profile -o prof1 -f true python jarvis_engine/prof_engine.py | Out-Null
& $n stats --force-export=true --report cuda_gpu_kern_sum --format csv prof1.nsys-rep | Select-Object -First 24 | Out-File prof1_kern.txt -Encoding utf8
