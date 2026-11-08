param([Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^(?:[A-Za-z0-9_-]+__)?\d{8}T\d{6}_[0-9a-f]{10}$') { throw 'Invalid run ID' }
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$row = @(Import-Csv -LiteralPath (Join-Path $projectRoot 'logs/run_archives.csv') | Where-Object run_id -eq $RunId)
if ($row.Count -ne 1) { throw 'Archive record not unique' }
$archive = Join-Path $projectRoot "archives/runs/$RunId.zip"
$destination = Join-Path $projectRoot "logs/runs/$RunId"
if (Test-Path -LiteralPath $destination) { throw 'Run already exists; never overwrite' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $row[0].archive_sha256) { throw 'Archive checksum mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$opened = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $opened.Entries) {
        $target = [IO.Path]::GetFullPath((Join-Path $destination $entry.FullName))
        if (-not $target.StartsWith($destination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive path' }
    }
} finally { $opened.Dispose() }
[IO.Compression.ZipFile]::ExtractToDirectory($archive, $destination)
Write-Output "Restored $RunId to $destination; archive retained"
