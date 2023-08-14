export is_greater

#   If two genes have the same expression value,
#   a random order is returned.
function is_greater(x::Number, y::Number,threshold_value::Number = 0.1)
	if abs(x - y) < threshold_value
		return rand(Bool)
	else
		return x > y
	end
end