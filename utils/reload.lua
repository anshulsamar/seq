-- reloads module for ease in interpreter

function reload(name)
   package.loaded[name] = nil
   require(name)
end
   