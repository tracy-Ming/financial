load = {}

require 'io'
require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

function load:info()
    print('started loading data')
end

function load:data_loading(args)
    self:info()
    local filePath = args.filepath or 'sample.txt'
    
    local i = 0
    for line in io.lines(filePath) do
      if i == 0 then
        COLS = #line:split(',')
      end
      i = i + 1
    end
    
    local ROWS = i - 1  -- Minus 1 because of header
    
    local file = io.open(filePath, 'r')
    local header = file:read()
    
    local data = torch.Tensor(ROWS, COLS-2)
    
    local i = 0
    for line in file:lines('*l') do
      i = i + 1
      local l = line:split(',')
      for key, val in ipairs(l) do
         if key>2 then 
            data[i][key-2] = val          
         end
      end
    end
    --print(data)
    file:close()
    return data,ROWS
end

--
--local filePath = 'sample.txt'
--
--local i = 0
--for line in io.lines(filePath) do
--  if i == 0 then
--    COLS = #line:split(',')
--  end
--  i = i + 1
--end
--
--local ROWS = i - 1 
--
--local data = torch.Tensor(ROWS, COLS)
--
--local file = io.open(filePath,'r')
--if file then
--    local index=0
--    local attributes={}
--    local instance={}
--    for line in file:lines() do
--    local instances={}
--           if index==0 then
--                attributes = line:split(',')
--                print(attributes)
--           else
--                instances=line:split(',')   
--                for x=2, #instances do
--                      print(instances)
--                end
--                instance[index]=instances
--           end 
--           index=index+1
--        --do something with that data
--    end
-- local d=torch.Tensor(instance)
--   print(d)
--else
--    print('File do not exists!')   
--end
--file:close()