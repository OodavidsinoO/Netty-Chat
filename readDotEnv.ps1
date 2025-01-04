[CmdletBinding(SupportsShouldProcess = $true)]
[OutputType([System.Object[]])]
param(
    [Parameter(Position=0, ValueFromPipeline)]
    [ValidateNotNullOrEmpty()]
    [string]
    $LiteralPath = (Join-Path -Path $PSScriptRoot -ChildPath '.env'),

    [Parameter()]
    [switch]
    $Force
)

$linecursor = 0

if (Test-Path -LiteralPath $LiteralPath) {
    Get-Content -LiteralPath $LiteralPath `
    | ForEach-Object { # go through line by line
        $line = $_.trim() # trim whitespace
        if ($line -match "^\s*#") {
            # it's a comment
            Write-Verbose -Message "Found comment $line at line $linecursor. discarding"
        }
        elseif ($line -match "^\s*$") {
            # it's a blank line
            Write-Verbose -Message "Found a blank line at line $linecursor, discarding"
        }
        elseif ($line -match "^\s*(?<key>[^\n\b\a\f\v\r\s]+)\s*=\s*(?<value>[^\n\b\a\f\v\r]*)$") {
            # it's not a comment, parse it
            # find the first '='
            $key = $Matches["key"]
            $value = $Matches["value"]

            Write-Verbose -Message "Found [$key] with value [$value]"

            # remove potential trailing comment
            if (-not [string]::IsNullOrWhiteSpace($value)) {
              $idx = $value.IndexOf('#')
              if (-1 -lt $idx) {
                Write-Verbose -Message "`tRemoving trailing comment"
                $value = $value.Substring(0, $idx - 1).trimEnd()
              }
              $quote = $value[0]
              if ($quote -in "`"", "`'") {
                Write-Verbose -Message "`tQuoted value found, trimming quotes"
                $value = $value.trim($quote)
                Write-Verbose -Message "`tValue is now [$value]"
              }
            }

            if ($Force -or -not [System.Environment]::GetEnvironmentVariable($key, 'Process')) {
                if ($PSCmdlet.ShouldProcess("Environment variable [$key]", "Set value [$value]")) {
                  [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
                }
            }
            else {
                Write-Verbose -Message "Environment variable $key already exists"
            }
        }
        else {
          Write-Warning "Invalid line $linecursor -> [$line]"
        }
        $linecursor++
    }
}
else {
    Write-Verbose "Dotenv file $LiteralPath not found."
}