$f = 'C:\Users\LENOVO\AppData\Local\Temp\devin.exe-overflows\shell-b3a6e2-abc2cd5a74c3b55f\content.txt'
while ($true) {
    Start-Sleep 20
    if (!(Test-Path $f)) { continue }
    $tail = Get-Content $f -Tail 60
    $tail | Where-Object { $_ -match 'state=' } | Select-Object -Last 1
    $tail | Where-Object { $_ -match 'submit_run|resubmit.*->|my_rank' } | Select-Object -Last 6
    if ($tail -match 'submit_run_versioned.*->\s*(200|530)') { break }
}
